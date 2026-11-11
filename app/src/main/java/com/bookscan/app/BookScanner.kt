package com.bookscan.app

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

class ScanResult(val pages: List<Bitmap>, val pxPerCm: Double)

enum class SplitMode { AUTO, ALWAYS, NEVER }

data class ScanOptions(
    val split: SplitMode = SplitMode.AUTO,
    val autoCrop: Boolean = true,
    val deskew: Boolean = true,
    val spreadWcm: Double = 30.0,     // rộng thật của sách MỞ (2 trang), cm
    val spreadHcm: Double = 21.0,     // cao thật của sách, cm
    val marginCm: Double = 0.5,       // lề dư quanh nội dung
    val merge: Boolean = false,       // sau khi xử lý riêng từng trang, ghép lại thành 1 ảnh 2 trang
    val maxSide: Int = 3000
)

/**
 * Ảnh chụp 2 trang (camera ngang, biết kích thước sách) -> tìm viền sách theo khối chữ + lề + mép giấy -> nắn phẳng -> tách theo gáy
 * -> xoá bóng -> đen trắng -> chỉnh nghiêng -> cắt theo nội dung + lề dư (mặc định 1 cm).
 */
object BookScanner {

    fun process(src: Bitmap, opt: ScanOptions): ScanResult {
        var img = bitmapToBgr(src)
        img = limit(img, opt.maxSide)

        var spread = img
        var ppc = img.cols() / opt.spreadWcm      // px/cm (ước lượng khi không có viền)
        var sized = false
        var quad: Array<Point>? = null
        if (opt.autoCrop) {
            // Thuật toán mới: dùng kích thước thật của sách (camera ngang) -> nắn phẳng đúng tỉ lệ cm.
            val q = try { EdgeDetector.detectQuad(img, opt.spreadWcm, opt.spreadHcm) } catch (e: Exception) { null }
            if (q != null) {
                val sw = warpSized(img, q, opt.spreadWcm, opt.spreadHcm)
                spread = sw.first
                ppc = sw.second
                sized = true
                quad = q
            } else {
                // dự phòng: không thấy chữ -> dùng bao quanh vùng sáng như bản cũ
                val q2 = detectQuad(img)
                if (q2 != null) {
                    spread = warp(img, q2)
                    ppc = spread.cols() / opt.spreadWcm
                }
            }
        }

        var ratio = spread.cols().toDouble() / spread.rows()
        if (!sized && opt.split == SplitMode.ALWAYS && ratio < 1.0) {
            val r = Mat()
            Core.rotate(spread, r, Core.ROTATE_90_CLOCKWISE)
            spread = r
            ratio = spread.cols().toDouble() / spread.rows()
            ppc = spread.cols() / opt.spreadWcm
        }
        val doSplit = when (opt.split) {
            SplitMode.ALWAYS -> true
            SplitMode.NEVER -> false
            SplitMode.AUTO -> ratio > 1.15
        }

        val pages = ArrayList<Mat>()
        if (doSplit) {
            val qd = quad
            if (sized && qd != null) {
                // mỗi trang một phép biến đổi phối cảnh riêng: tứ giác (mép ngoài + đường gáy) -> chữ nhật
                pages.addAll(splitSized(img, qd, spread, ppc))
            } else {
                val gx = findGutter(spread)
                pages.add(spread.submat(0, spread.rows(), 0, gx))
                pages.add(spread.submat(0, spread.rows(), gx, spread.cols()))
            }
        } else {
            pages.add(spread)
        }
        // xử lý RIÊNG từng trang trước (xoá bóng, đen trắng, chỉnh nghiêng, cắt theo nội dung)
        val cleaned = pages.map { cleanPage(it, opt, ppc) }
        val outs = if (opt.merge && cleaned.size == 2) listOf(mergeSideBySide(cleaned[0], cleaned[1])) else cleaned
        return ScanResult(outs.map { toBitmap(it) }, ppc)
    }

    /** Ghép trang trái + trang phải (cùng px/cm) thành 1 ảnh: cao bằng nhau (đệm trắng), sát nhau như sách mở. */
    private fun mergeSideBySide(a: Mat, b: Mat): Mat {
        val hh = max(a.rows(), b.rows())
        fun pad(m: Mat): Mat {
            val top = (hh - m.rows()) / 2
            val out = Mat()
            Core.copyMakeBorder(m, out, top, hh - m.rows() - top, 0, 0, Core.BORDER_CONSTANT, Scalar(255.0))
            return out
        }
        val out = Mat()
        Core.hconcat(listOf(pad(a), pad(b)), out)
        return out
    }

    // ---------- tiện ích ----------
    private fun bitmapToBgr(b: Bitmap): Mat {
        val rgba = Mat()
        Utils.bitmapToMat(b, rgba)
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        return bgr
    }

    private fun toBitmap(m: Mat): Bitmap {
        val rgba = Mat()
        Imgproc.cvtColor(
            m, rgba,
            if (m.channels() == 1) Imgproc.COLOR_GRAY2RGBA else Imgproc.COLOR_BGR2RGBA
        )
        val bmp = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgba, bmp)
        rgba.release()
        return bmp
    }

    private fun limit(m: Mat, maxSide: Int): Mat {
        val big = max(m.cols(), m.rows())
        if (big <= maxSide) return m
        val s = maxSide.toDouble() / big
        val out = Mat()
        Imgproc.resize(m, out, Size(), s, s, Imgproc.INTER_AREA)
        m.release()
        return out
    }

    // ---------- 1. tìm viền sách ----------
    private fun detectQuad(bgr: Mat): Array<Point>? {
        val scale = 800.0 / max(bgr.cols(), bgr.rows())
        val small = Mat()
        Imgproc.resize(bgr, small, Size(), scale, scale, Imgproc.INTER_AREA)
        val gray = Mat()
        Imgproc.cvtColor(small, gray, Imgproc.COLOR_BGR2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(7.0, 7.0), 0.0)
        val bin = Mat()
        Imgproc.threshold(gray, bin, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(15.0, 15.0))
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_CLOSE, k)
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_OPEN, k)

        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(bin, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val imgArea = small.cols().toDouble() * small.rows()
        val center = Point(small.cols() / 2.0, small.rows() / 2.0)
        var best: MatOfPoint? = null
        var bestScore = 0.0
        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < 0.2 * imgArea) continue
            val inside = Imgproc.pointPolygonTest(MatOfPoint2f(*c.toArray()), center, false) >= 0
            val score = area * (if (inside) 1.0 else 0.3)
            if (score > bestScore) { bestScore = score; best = c }
        }
        if (best == null) return null

        val pts = best.toArray()
        val hullIdx = MatOfInt()
        Imgproc.convexHull(best, hullIdx)
        val hull = hullIdx.toArray().map { pts[it] }.toTypedArray()
        val hull2f = MatOfPoint2f(*hull)
        val peri = Imgproc.arcLength(hull2f, true)

        var quad: Array<Point>? = null
        var eps = 0.01
        while (eps <= 0.12) {
            val ap = MatOfPoint2f()
            Imgproc.approxPolyDP(hull2f, ap, eps * peri, true)
            if (ap.rows() == 4) { quad = ap.toArray(); break }
            if (ap.rows() < 4) break
            eps += 0.005
        }
        if (quad == null) {
            val r = Imgproc.minAreaRect(hull2f)
            val p = Array(4) { Point() }
            r.points(p)
            quad = p
        }
        // xấp xỉ 4 đỉnh hay "ăn" vào góc -> mở rộng để bao trọn toàn bộ đường viền
        val ordered = orderPoints(quad)
        val maxShift = 0.15 * hypot(small.cols().toDouble(), small.rows().toDouble())
        val enclosing = enclosingQuad(ordered, hull, maxShift)
        val scaled = enclosing.map { Point(it.x / scale, it.y / scale) }.toTypedArray()
        return scaled
    }

    /** trả về [tl, tr, br, bl] */
    private fun orderPoints(p: Array<Point>): Array<Point> {
        val tl = p.minByOrNull { it.x + it.y }!!
        val br = p.maxByOrNull { it.x + it.y }!!
        val tr = p.minByOrNull { it.y - it.x }!!
        val bl = p.maxByOrNull { it.y - it.x }!!
        return arrayOf(tl, tr, br, bl)
    }

    /** Dịch 4 cạnh ra ngoài tới khi mọi điểm của hull nằm trong, rồi giao các cạnh lại thành 4 góc mới. */
    private fun enclosingQuad(q: Array<Point>, hull: Array<Point>, maxShift: Double): Array<Point> {
        val a = DoubleArray(4); val b = DoubleArray(4); val c = DoubleArray(4)
        for (i in 0 until 4) {
            val p = q[i]; val r = q[(i + 1) % 4]
            val dx = r.x - p.x; val dy = r.y - p.y
            val len = hypot(dx, dy)
            if (len < 1e-6) return q
            val nx = dy / len; val ny = -dx / len   // pháp tuyến hướng ra ngoài (tl->tr->br->bl)
            var cc = nx * p.x + ny * p.y
            for (h in hull) cc = max(cc, nx * h.x + ny * h.y)
            a[i] = nx; b[i] = ny; c[i] = cc
        }
        val out = Array(4) { q[it] }
        for (i in 0 until 4) {
            val j = (i + 3) % 4 // cạnh kết thúc tại góc i
            val det = a[j] * b[i] - a[i] * b[j]
            if (abs(det) < 1e-6) continue
            val x = (c[j] * b[i] - c[i] * b[j]) / det
            val y = (a[j] * c[i] - a[i] * c[j]) / det
            if (hypot(x - q[i].x, y - q[i].y) <= maxShift) out[i] = Point(x, y)
        }
        return out
    }

    private fun warp(bgr: Mat, q: Array<Point>): Mat {
        val tl = q[0]; val tr = q[1]; val br = q[2]; val bl = q[3]
        val w = max(hypot(tr.x - tl.x, tr.y - tl.y), hypot(br.x - bl.x, br.y - bl.y)).toInt().coerceAtLeast(200)
        val h = max(hypot(bl.x - tl.x, bl.y - tl.y), hypot(br.x - tr.x, br.y - tr.y)).toInt().coerceAtLeast(200)
        val src = MatOfPoint2f(tl, tr, br, bl)
        val dst = MatOfPoint2f(
            Point(0.0, 0.0), Point(w - 1.0, 0.0), Point(w - 1.0, h - 1.0), Point(0.0, h - 1.0)
        )
        val m = Imgproc.getPerspectiveTransform(src, dst)
        val out = Mat()
        Imgproc.warpPerspective(bgr, out, m, Size(w.toDouble(), h.toDouble()),
            Imgproc.INTER_CUBIC, Core.BORDER_REPLICATE)
        return out
    }

    /**
     * Nắn phẳng đúng tỉ lệ cm: ảnh ra rộng Sw*ppc, cao Sh*ppc (px/cm giữ ≤ 300 dpi, không phóng quá độ phân giải gốc).
     * Trả về (ảnh, px/cm).
     */
    private fun warpSized(bgr: Mat, q: Array<Point>, swCm: Double, shCm: Double): Pair<Mat, Double> {
        val tl = q[0]; val tr = q[1]; val br = q[2]; val bl = q[3]
        val pxW = (hypot(tr.x - tl.x, tr.y - tl.y) + hypot(br.x - bl.x, br.y - bl.y)) / 2
        val pxH = (hypot(bl.x - tl.x, bl.y - tl.y) + hypot(br.x - tr.x, br.y - tr.y)) / 2
        val native = Math.sqrt((pxW / swCm) * (pxH / shCm))
        val ppc = min(native, 300.0 / 2.54).coerceAtLeast(20.0)
        val w = Math.round(swCm * ppc).toInt()
        val h = Math.round(shCm * ppc).toInt()
        val src = MatOfPoint2f(tl, tr, br, bl)
        val dst = MatOfPoint2f(
            Point(0.0, 0.0), Point(w - 1.0, 0.0), Point(w - 1.0, h - 1.0), Point(0.0, h - 1.0)
        )
        val m = Imgproc.getPerspectiveTransform(src, dst)
        val out = Mat()
        Imgproc.warpPerspective(bgr, out, m, Size(w.toDouble(), h.toDouble()),
            Imgproc.INTER_CUBIC, Core.BORDER_REPLICATE)
        return out to ppc
    }

    /**
     * Tách 2 trang: BƯỚC 1 cắt thô từng trang kèm nền (tứ giác = mép ngoài + đường gáy, nới 1,5 cm ra ngoài),
     * BƯỚC 2 từng trang tự dò lại mép giấy rồi nắn bằng homography riêng về chữ nhật.
     * Gáy dò ở nửa trên và nửa dưới của ảnh nắn tạm rồi chiếu ngược về ảnh gốc,
     * nên cả hai trang ra đúng cùng kích thước chữ nhật dù phối cảnh/cong làm hai nửa to nhỏ khác nhau.
     */
    private fun splitSized(img: Mat, q: Array<Point>, spread: Mat, ppc: Double): List<Mat> {
        val w = spread.cols()
        val h = spread.rows()
        val dst = MatOfPoint2f(Point(0.0, 0.0), Point(w - 1.0, 0.0), Point(w - 1.0, h - 1.0), Point(0.0, h - 1.0))
        val inv = Imgproc.getPerspectiveTransform(dst, MatOfPoint2f(*q))   // ảnh nắn tạm -> ảnh gốc
        val h2 = h / 2
        val lo = w * 0.44
        val hi = w * 0.56
        var xt = EdgeDetector.findCut(spread.submat(0, h2, 0, w)).x.toDouble().coerceIn(lo, hi)
        var xb = EdgeDetector.findCut(spread.submat(h2, h, 0, w)).x.toDouble().coerceIn(lo, hi)
        if (abs(xt - xb) > 0.04 * w) { val m = (xt + xb) / 2; xt = m; xb = m }
        val k = (xb - xt) / (h / 2.0)
        val gIn = MatOfPoint2f(Point(xt - k * h / 4, 0.0), Point(xb + k * h / 4, h - 1.0))
        val gOut = MatOfPoint2f()
        Core.perspectiveTransform(gIn, gOut, inv)
        val g = gOut.toArray()
        val pw = w / 2
        val d2 = MatOfPoint2f(Point(0.0, 0.0), Point(pw - 1.0, 0.0), Point(pw - 1.0, h - 1.0), Point(0.0, h - 1.0))
        val rough = listOf(
            arrayOf(q[0], g[0], g[1], q[3]),    // trang trái
            arrayOf(g[0], q[1], q[2], g[1])     // trang phải
        )
        val m = Math.round(1.5 * ppc).toInt()   // dư nền 1,5 cm quanh mép ngoài
        val gm = Math.round(0.2 * ppc).toInt()  // phía gáy chỉ dư chút xíu
        val out = ArrayList<Mat>()
        for ((side, rq) in rough.withIndex()) {
            // BƯỚC 1: cắt thô từng trang KÈM NỀN (nắn gần đúng, trang nằm ở khung inner trong crop)
            val x0 = if (side == 0) m else gm
            val cw = m + pw + gm
            val ch = h + 2 * m
            val dstIn = MatOfPoint2f(
                Point(x0.toDouble(), m.toDouble()), Point(x0 + pw - 1.0, m.toDouble()),
                Point(x0 + pw - 1.0, m + h - 1.0), Point(x0.toDouble(), m + h - 1.0)
            )
            val srcQ = MatOfPoint2f(*rq)
            val mm = Imgproc.getPerspectiveTransform(srcQ, dstIn)
            val crop = Mat()
            Imgproc.warpPerspective(img, crop, mm, Size(cw.toDouble(), ch.toDouble()),
                Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE)
            // BƯỚC 2: riêng trang này dò lại 3 mép ngoài (trên/dưới/ngoài) rồi nắn về chữ nhật
            val outer = if (side == 0) setOf(0, 2, 3) else setOf(0, 1, 2)
            val q2 = try {
                EdgeDetector.refinePage(crop, doubleArrayOf(x0.toDouble(), m.toDouble(), (x0 + pw).toDouble(), (m + h).toDouble()), ppc, outer)
            } catch (e: Exception) { null }
            crop.release()
            val finalQ: MatOfPoint2f = if (q2 != null) {
                val back = MatOfPoint2f()
                Core.perspectiveTransform(MatOfPoint2f(*q2), back, mm.inv())
                back
            } else srcQ
            val m2 = Imgproc.getPerspectiveTransform(finalQ, d2)
            val page = Mat()
            Imgproc.warpPerspective(img, page, m2, Size(pw.toDouble(), h.toDouble()),
                Imgproc.INTER_CUBIC, Core.BORDER_REPLICATE)
            out.add(page)
        }
        return out
    }

    // ---------- 2. rãnh gáy ----------
    private fun findGutter(spread: Mat): Int {
        val w = spread.cols()
        val gray = Mat()
        Imgproc.cvtColor(spread, gray, Imgproc.COLOR_BGR2GRAY)
        val col = Mat()
        Core.reduce(gray, col, 0, Core.REDUCE_AVG, CvType.CV_32F)
        var k = w / 40
        if (k % 2 == 0) k++
        Imgproc.GaussianBlur(col, col, Size(max(k, 5).toDouble(), 1.0), 0.0)
        val a = FloatArray(w)
        col.get(0, 0, a)
        val lo = (w * 0.42).toInt()
        val hi = (w * 0.58).toInt()
        var bi = w / 2
        var bv = Float.MAX_VALUE
        for (x in lo..hi) if (a[x] < bv) { bv = a[x]; bi = x }
        val median = a.sorted()[w / 2]
        gray.release(); col.release()
        return if (bv > 0.97f * median) w / 2 else bi
    }

    // ---------- 3. làm sạch trang -> đen trắng ----------
    private fun cleanPage(page: Mat, opt: ScanOptions, pxPerCm: Double): Mat {
        val ix = (page.cols() * 0.008).toInt()
        val iy = (page.rows() * 0.008).toInt()
        val roi = page.submat(iy, page.rows() - iy, ix, page.cols() - ix).clone()

        val g = Mat()
        Imgproc.cvtColor(roi, g, Imgproc.COLOR_BGR2GRAY)
        roi.release()
        val flat = flatten(g)

        val lo = min(percentile(flat, 0.01), 110)
        val hi = 235
        val lut = Mat(1, 256, CvType.CV_8U)
        val tab = ByteArray(256) { i ->
            val t = ((i - lo).toDouble() / (hi - lo)).coerceIn(0.0, 1.0)
            (255 * t).toInt().toByte()
        }
        lut.put(0, 0, tab)
        var gray = Mat()
        Core.LUT(flat, lut, gray)

        if (opt.deskew) {
            val ang = estimateSkew(gray)
            if (abs(ang) > 0.15) gray = rotate(gray, ang)
        }

        val bw = Mat()
        Imgproc.threshold(gray, bw, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)

        return cropToContent(bw, (opt.marginCm * pxPerCm).toInt().coerceAtLeast(0))
    }

    /**
     * Bỏ các mảng đen chạm mép ảnh (bóng bàn/mép giấy), tìm khung nội dung,
     * rồi cắt khung đó cộng thêm lề [marginPx]; chỗ thiếu được đệm trắng.
     */
    private fun cropToContent(bw: Mat, marginPx: Int): Mat {
        val W = bw.cols(); val H = bw.rows()
        val ink = Mat()
        Core.bitwise_not(bw, ink)
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        Imgproc.connectedComponentsWithStats(ink, labels, stats, cents, 8, CvType.CV_32S)
        val n = stats.rows()
        val st = IntArray(n * 5)
        stats.get(0, 0, st)

        val minArea = max(8, (W.toLong() * H * 3e-6).toInt())
        val lefts = ArrayList<Int>(); val tops = ArrayList<Int>()
        val rights = ArrayList<Int>(); val bottoms = ArrayList<Int>()
        val remove = ArrayList<Int>()
        for (i in 1 until n) {
            val l = st[i * 5]; val t = st[i * 5 + 1]
            val w = st[i * 5 + 2]; val h = st[i * 5 + 3]; val a = st[i * 5 + 4]
            val touches = l <= 1 || t <= 1 || l + w >= W - 1 || t + h >= H - 1
            if (touches) remove.add(i)
            else if (a >= minArea) {
                lefts.add(l); tops.add(t); rights.add(l + w); bottoms.add(t + h)
            }
        }
        if (remove.isNotEmpty()) {
            val m = Mat()
            for (id in remove.take(300)) {
                Core.compare(labels, Scalar(id.toDouble()), m, Core.CMP_EQ)
                bw.setTo(Scalar(255.0), m)
            }
            m.release()
        }
        ink.release(); labels.release(); stats.release(); cents.release()

        if (lefts.size < 5) return bw

        val q = min(1, (lefts.size * 0.001).toInt())
        val left = lefts.sorted()[q]
        val top = tops.sorted()[q]
        val right = rights.sorted()[rights.size - 1 - q]
        val bottom = bottoms.sorted()[bottoms.size - 1 - q]

        val m = marginPx
        val padded = Mat()
        Core.copyMakeBorder(bw, padded, m, m, m, m, Core.BORDER_CONSTANT, Scalar(255.0))
        // toạ độ trong ảnh đệm = toạ độ gốc + m
        val x0 = left
        val y0 = top
        val x1 = min(padded.cols(), right + 2 * m)
        val y1 = min(padded.rows(), bottom + 2 * m)
        if (x1 - x0 < 50 || y1 - y0 < 50) return bw
        return padded.submat(y0, y1, x0, x1).clone()
    }

    /** chia ảnh cho nền (ước lượng bằng dilate) để xoá bóng và làm trắng giấy */
    private fun flatten(ch: Mat): Mat {
        val s = 0.25
        val small = Mat()
        Imgproc.resize(ch, small, Size(), s, s, Imgproc.INTER_AREA)
        var ks = max(small.cols(), small.rows()) / 18
        if (ks % 2 == 0) ks++
        ks = max(ks, 9)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(ks.toDouble(), ks.toDouble()))
        val bgS = Mat()
        Imgproc.dilate(small, bgS, kernel)
        Imgproc.GaussianBlur(bgS, bgS, Size(ks.toDouble(), ks.toDouble()), 0.0)
        val bg = Mat()
        Imgproc.resize(bgS, bg, ch.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val out = Mat()
        Core.divide(ch, bg, out, 255.0)
        small.release(); bgS.release(); bg.release()
        return out
    }

    private fun percentile(g: Mat, p: Double): Int {
        val s = Mat()
        Imgproc.resize(g, s, Size(256.0, 256.0), 0.0, 0.0, Imgproc.INTER_AREA)
        val b = ByteArray(256 * 256)
        s.get(0, 0, b)
        val h = IntArray(256)
        for (v in b) h[v.toInt() and 0xFF]++
        val target = (b.size * p).toInt()
        var acc = 0
        for (i in 0..255) { acc += h[i]; if (acc >= target) return i }
        return 0
    }

    // ---------- 4. chỉnh nghiêng ----------
    private fun estimateSkew(g: Mat): Double {
        val sc = 500.0 / g.cols()
        val small = Mat()
        Imgproc.resize(g, small, Size(), sc, sc, Imgproc.INTER_AREA)
        val mask = Mat()
        Imgproc.threshold(small, mask, 0.0, 255.0, Imgproc.THRESH_BINARY_INV or Imgproc.THRESH_OTSU)
        val mx = (mask.cols() * 0.06).toInt()
        val my = (mask.rows() * 0.06).toInt()
        val inner = mask.submat(my, mask.rows() - my, mx, mask.cols() - mx).clone()
        val center = Point(inner.cols() / 2.0, inner.rows() / 2.0)
        var best = 0.0
        var bestScore = -1.0
        val rows = Mat()
        val rot = Mat()
        for (a in -40..40) {
            val angle = a * 0.125
            val m = Imgproc.getRotationMatrix2D(center, angle, 1.0)
            Imgproc.warpAffine(inner, rot, m, inner.size(), Imgproc.INTER_LINEAR)
            Core.reduce(rot, rows, 1, Core.REDUCE_SUM, CvType.CV_32F)
            val arr = FloatArray(rows.rows())
            rows.get(0, 0, arr)
            var score = 0.0
            for (i in 1 until arr.size) { val d = (arr[i] - arr[i - 1]).toDouble(); score += d * d }
            if (score > bestScore) { bestScore = score; best = angle }
        }
        return best
    }

    private fun rotate(img: Mat, angle: Double): Mat {
        val m = Imgproc.getRotationMatrix2D(Point(img.cols() / 2.0, img.rows() / 2.0), angle, 1.0)
        val out = Mat()
        Imgproc.warpAffine(img, out, m, img.size(), Imgproc.INTER_CUBIC,
            Core.BORDER_CONSTANT, Scalar(255.0))
        return out
    }
}
