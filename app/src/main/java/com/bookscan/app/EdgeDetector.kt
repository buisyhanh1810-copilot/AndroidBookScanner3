package com.bookscan.app

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
import java.util.Random
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Phát hiện viền sách (4 góc) và vết cắt gáy, DÙNG KÍCH THƯỚC THẬT của sách (cm) do người dùng nhập.
 * Giả định: camera nằm ngang, sách mở nằm ngang trong khung (trang trái – gáy – trang phải).
 *
 * Ý tưởng (đã kiểm định trên bộ ảnh tổng hợp có đáp án, xem tools/prototype):
 *  1. Khối chữ thân văn bản (bỏ vệt mép giấy, bóng, chữ lẻ) -> 4 đường biên khối chữ (bao lồi).
 *  2. Từ kích thước sách đã biết suy ra tỉ lệ px/cm và vị trí DỰ ĐOÁN mép giấy = biên chữ + lề điển hình.
 *  3. Mỗi cạnh được "bắt dính" vào mép giấy thật: quét 28 tia vuông góc quanh vị trí dự đoán,
 *     tín hiệu = bước nhảy độ sáng (nhiều tỉ lệ, chịu mép mờ) + bước nhảy sắc độ (giấy so với nền),
 *     cổng chữ (không cắt vào chữ), rồi RANSAC có trọng số. Cạnh không đủ tin cậy giữ vị trí dự đoán.
 *  4. Ràng buộc an toàn: mọi nét chữ nằm trong tứ giác.
 *  5. Gáy: bóng gáy + khoảng trống chữ + gần giữa, kẹp vào khoảng trống và đặt gần tâm khoảng trống.
 */
object EdgeDetector {

    class Cut(val x: Int, val risky: Boolean)

    private class V(val x: Double, val y: Double) {
        operator fun plus(o: V) = V(x + o.x, y + o.y)
        operator fun minus(o: V) = V(x - o.x, y - o.y)
        operator fun times(k: Double) = V(x * k, y * k)
        operator fun unaryMinus() = V(-x, -y)
        infix fun dot(o: V) = x * o.x + y * o.y
        fun len() = hypot(x, y)
    }

    /** Đường thẳng: điểm p, hướng d (đơn vị), pháp tuyến o hướng RA NGOÀI. */
    private class Ln(val p: V, val d: V, val o: V)

    // ================================================================== tiện ích
    private fun odd(n: Double): Int {
        val i = n.toInt()
        return if (i % 2 == 1) i else i + 1
    }

    private fun floats(m: Mat): FloatArray {
        val a = FloatArray(m.rows() * m.cols())
        m.get(0, 0, a)
        return a
    }

    private fun clip(v: Double, lo: Double, hi: Double) = if (v < lo) lo else if (v > hi) hi else v

    /** Phân vị p (0..100) theo nội suy tuyến tính như numpy. */
    private fun pct(a: DoubleArray, p: Double): Double {
        if (a.isEmpty()) return 0.0
        val s = a.sortedArray()
        val idx = p / 100.0 * (s.size - 1)
        val lo = floor(idx).toInt()
        val hi = min(lo + 1, s.size - 1)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)
    }

    private fun median(a: DoubleArray) = pct(a, 50.0)

    private fun bil(a: FloatArray, w: Int, h: Int, x0: Double, y0: Double): Double {
        val x = clip(x0, 0.0, w - 1.001)
        val y = clip(y0, 0.0, h - 1.001)
        val xi = x.toInt()
        val yi = y.toInt()
        val fx = x - xi
        val fy = y - yi
        val i = yi * w + xi
        return a[i] * (1 - fx) * (1 - fy) + a[i + 1] * fx * (1 - fy) +
            a[i + w] * (1 - fx) * fy + a[i + w + 1] * fx * fy
    }

    private fun orderV(p: List<V>): Array<V> {
        val tl = p.minByOrNull { it.x + it.y }!!
        val br = p.maxByOrNull { it.x + it.y }!!
        val tr = p.minByOrNull { it.y - it.x }!!
        val bl = p.maxByOrNull { it.y - it.x }!!
        return arrayOf(tl, tr, br, bl)
    }

    private fun area(q: Array<V>): Double {
        var s = 0.0
        for (i in q.indices) {
            val a = q[i]
            val b = q[(i + 1) % q.size]
            s += a.x * b.y - b.x * a.y
        }
        return abs(s) / 2.0
    }

    private fun isConvex(q: Array<V>): Boolean {
        var sign = 0
        for (i in q.indices) {
            val a = q[i]
            val b = q[(i + 1) % 4]
            val c = q[(i + 2) % 4]
            val cr = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            val s = if (cr > 0) 1 else if (cr < 0) -1 else 0
            if (s == 0) return false
            if (sign == 0) sign = s else if (s != sign) return false
        }
        return true
    }

    private fun inter(l1: Ln, l2: Ln): V? {
        val det = -l1.d.x * l2.d.y + l2.d.x * l1.d.y
        if (abs(det) < 1e-6) return null
        val r = l2.p - l1.p
        val t = (r.x * (-l2.d.y) - (-l2.d.x) * r.y) / det
        return l1.p + l1.d * t
    }

    private fun quadFromLines(l: List<Ln>): Array<V>? {
        val q = ArrayList<V>()
        for (i in 0 until 4) q.add(inter(l[(i + 3) % 4], l[i]) ?: return null)
        if (q.any { !it.x.isFinite() || !it.y.isFinite() }) return null
        return orderV(q)
    }

    private fun projOf(xs: DoubleArray, ys: DoubleArray, o: V) =
        DoubleArray(xs.size) { xs[it] * o.x + ys[it] * o.y }

    // ================================================================== mặt nạ nét chữ
    /** Blackhat + ngưỡng có trễ. [cx0,cx1) (vùng gáy) dùng ngưỡng yếu thấp hơn để bắt chữ mờ. */
    private fun textMask(gray: Mat, cx0: Int = -1, cx1: Int = -1): Mat {
        val h = gray.rows()
        val w = gray.cols()
        val k = odd(max(9.0, 0.025 * max(h, w)))
        val kern = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k.toDouble(), k.toDouble()))
        val bh = Mat()
        Imgproc.morphologyEx(gray, bh, Imgproc.MORPH_BLACKHAT, kern)
        val tmp = Mat()
        var thr = Imgproc.threshold(bh, tmp, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
        thr = max(thr, 14.0)
        val strong = Mat()
        Imgproc.threshold(bh, strong, thr, 1.0, Imgproc.THRESH_BINARY)
        val weak = Mat()
        Imgproc.threshold(bh, weak, max(8.0, 0.5 * thr), 1.0, Imgproc.THRESH_BINARY)
        if (cx0 >= 0 && cx1 > cx0) {
            val wk2 = Mat()
            Imgproc.threshold(bh, wk2, max(6.0, 0.35 * thr), 1.0, Imgproc.THRESH_BINARY)
            wk2.submat(0, h, cx0, cx1).copyTo(weak.submat(0, h, cx0, cx1))
            wk2.release()
        }
        val near = Mat()
        Imgproc.dilate(strong, near, Mat.ones(15, 15, CvType.CV_8U))
        val out = Mat()
        Core.bitwise_and(weak, near, out)
        bh.release(); tmp.release(); strong.release(); weak.release(); near.release()
        return out
    }

    /** Điểm nét chữ THÂN VĂN BẢN: nét cỡ chữ + mật độ cục bộ cao (loại vệt mép giấy, bóng, chữ lẻ). */
    private class TextPts(val xs: DoubleArray, val ys: DoubleArray) {
        val n get() = xs.size
    }

    private fun textPoints(l8: Mat, w: Int, h: Int, long: Double): TextPts {
        val tm = textMask(l8)
        val lab0 = Mat()
        val st0m = Mat()
        val c0 = Mat()
        val n0 = Imgproc.connectedComponentsWithStats(tm, lab0, st0m, c0, 8, CvType.CV_32S)
        val st0 = IntArray(n0 * 5)
        st0m.get(0, 0, st0)
        val ok0 = BooleanArray(n0)
        for (i in 1 until n0) {
            val cw = st0[i * 5 + 2]
            val ch = st0[i * 5 + 3]
            val ar = st0[i * 5 + 4]
            if (cw <= 0.07 * long && ch <= 0.035 * long && ar >= 3 &&
                max(cw, ch).toDouble() / max(1, min(cw, ch)) < 12
            ) ok0[i] = true
        }
        val labA = IntArray(w * h)
        lab0.get(0, 0, labA)
        val tmB = ByteArray(w * h)
        for (i in 0 until w * h) tmB[i] = if (ok0[labA[i]]) 1 else 0
        val tm2 = Mat(h, w, CvType.CV_8U)
        tm2.put(0, 0, tmB)

        val kd = odd(max(15.0, 0.03 * long))
        val tmf = Mat()
        tm2.convertTo(tmf, CvType.CV_32F)
        val dens = Mat()
        Imgproc.boxFilter(tmf, dens, -1, Size(kd.toDouble(), kd.toDouble()))
        val core = Mat()
        Imgproc.threshold(dens, core, 0.10, 1.0, Imgproc.THRESH_BINARY)
        val core8 = Mat()
        core.convertTo(core8, CvType.CV_8U)
        Imgproc.morphologyEx(
            core8, core8, Imgproc.MORPH_CLOSE,
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(odd(0.03 * w).toDouble(), odd(0.02 * h).toDouble()))
        )
        val lab1 = Mat()
        val st1m = Mat()
        val c1 = Mat()
        val n1 = Imgproc.connectedComponentsWithStats(core8, lab1, st1m, c1, 8, CvType.CV_32S)
        val st1 = IntArray(n1 * 5)
        st1m.get(0, 0, st1)
        val keep = BooleanArray(n1)
        for (i in 1 until n1) keep[i] = st1[i * 5 + 4] >= 0.02 * h * w
        val labB = IntArray(w * h)
        lab1.get(0, 0, labB)
        val xs = ArrayList<Double>()
        val ys = ArrayList<Double>()
        for (i in 0 until w * h) {
            if (tmB[i].toInt() != 0 && keep[labB[i]]) {
                xs.add((i % w).toDouble()); ys.add((i / w).toDouble())
            }
        }
        val stride = max(1, xs.size / 8000 + (if (xs.size % 8000 != 0) 1 else 0))
        val sx = ArrayList<Double>()
        val sy = ArrayList<Double>()
        var k = 0
        while (k < xs.size) { sx.add(xs[k]); sy.add(ys[k]); k += stride }
        tm.release(); lab0.release(); st0m.release(); c0.release(); tm2.release(); tmf.release()
        dens.release(); core.release(); core8.release(); lab1.release(); st1m.release(); c1.release()
        return TextPts(sx.toDoubleArray(), sy.toDoubleArray())
    }

    // ================================================================== 4 đường biên khối chữ
    private fun hullOf(tp: TextPts): List<V> {
        val pts = Array(tp.n) { Point(tp.xs[it], tp.ys[it]) }
        val mp = MatOfPoint(*pts)
        val idx = MatOfInt()
        Imgproc.convexHull(mp, idx)
        val arr = mp.toArray()
        return idx.toArray().map { V(arr[it].x, arr[it].y) }
    }

    private fun boxOf(pts: List<V>): Array<V> {
        val m = MatOfPoint2f(*pts.map { Point(it.x, it.y) }.toTypedArray())
        val r = Imgproc.minAreaRect(m)
        val p = Array(4) { Point() }
        r.points(p)
        return orderV(p.map { V(it.x, it.y) })
    }

    private fun maxProj(tp: TextPts, o: V, p: Double): Double = pct(projOf(tp.xs, tp.ys, o), p)

    /** A: bao lồi -> xấp xỉ 4 đỉnh -> đẩy mỗi cạnh ra ngoài tới khi chứa mọi điểm chữ. */
    private fun linesA(tp: TextPts, hull: List<V>): List<Ln>? {
        val h2f = MatOfPoint2f(*hull.map { Point(it.x, it.y) }.toTypedArray())
        val peri = Imgproc.arcLength(h2f, true)
        var quad: Array<V>? = null
        var eps = 0.01
        while (eps <= 0.15) {
            val ap = MatOfPoint2f()
            Imgproc.approxPolyDP(h2f, ap, eps * peri, true)
            if (ap.rows() == 4) { quad = ap.toArray().map { V(it.x, it.y) }.toTypedArray(); break }
            if (ap.rows() < 4) break
            eps += 0.005
        }
        val box = orderV((quad ?: boxOf(hull)).toList())
        val cen = V(box.sumOf { it.x } / 4, box.sumOf { it.y } / 4)
        val out = ArrayList<Ln>()
        for (i in 0 until 4) {
            val a = box[i]
            val b = box[(i + 1) % 4]
            val len = (b - a).len()
            if (len < 1e-6) return null
            val d = (b - a) * (1.0 / len)
            var o = V(d.y, -d.x)
            if ((o dot ((a + b) * 0.5 - cen)) < 0) o = -o
            val proj = DoubleArray(tp.n) { (tp.xs[it] - a.x) * o.x + (tp.ys[it] - a.y) * o.y }
            val off = pct(proj, 99.8)
            out.add(Ln(a + o * max(off, 0.0), d, o))
        }
        return out
    }

    /** B: hướng mỗi cạnh = trung bình (theo độ dài) các cạnh bao lồi gần hướng hộp nhỏ nhất; vị trí = đường tựa ngoài cùng. */
    private fun linesB(tp: TextPts, hull: List<V>): List<Ln>? {
        val box = boxOf(hull)
        val boxC = V(box.sumOf { it.x } / 4, box.sumOf { it.y } / 4)
        val cen = V(hull.sumOf { it.x } / hull.size, hull.sumOf { it.y } / hull.size)
        val n = hull.size
        val out = ArrayList<Ln>()
        for (i in 0 until 4) {
            val a = box[i]
            val b = box[(i + 1) % 4]
            val l0 = (b - a).len()
            if (l0 < 1e-6) return null
            val dd = (b - a) * (1.0 / l0)
            var o0 = V(dd.y, -dd.x)
            if ((o0 dot ((a + b) * 0.5 - boxC)) < 0) o0 = -o0
            val d0 = V(-o0.y, o0.x)
            var num = 0.0
            var den = 0.0
            for (j in 0 until n) {
                val p = hull[j]
                val q = hull[(j + 1) % n]
                var e = q - p
                val len = e.len()
                if (len < 1e-3) continue
                e = e * (1.0 / len)
                var nn = V(e.y, -e.x)
                if ((nn dot ((p + q) * 0.5 - cen)) < 0) nn = -nn
                val ang = acos(clip(nn dot o0, -1.0, 1.0))
                if (ang < Math.toRadians(20.0)) {
                    if ((e dot d0) < 0) e = -e
                    val th = atan2(d0.x * e.y - d0.y * e.x, d0 dot e)
                    num += len * th; den += len
                }
            }
            val th = if (den > 0) num / den else 0.0
            val c = cos(th)
            val s = sin(th)
            val d = V(c * d0.x - s * d0.y, s * d0.x + c * d0.y)
            var o = V(d.y, -d.x)
            if ((o dot o0) < 0) o = -o
            val off = maxProj(tp, o, 99.8)
            out.add(Ln(o * off, d, o))
        }
        return out
    }

    /** Độ ôm sát: trung bình theo đoạn dọc cạnh của khoảng cách từ đường tới điểm chữ gần nhất. */
    private fun tightness(tp: TextPts, ln: Ln, long: Double, nb: Int = 24): Double {
        val pd = projOf(tp.xs, tp.ys, ln.d)
        val po = projOf(tp.xs, tp.ys, ln.o)
        val base = ln.p dot ln.o
        val lo = pct(pd, 5.0)
        val hi = pct(pd, 95.0)
        val cap = 0.1 * long
        val mins = DoubleArray(nb) { Double.MAX_VALUE }
        val cnt = IntArray(nb)
        val step = (hi - lo) / nb
        if (step <= 0) return 1e9
        for (i in pd.indices) {
            if (pd[i] < lo || pd[i] >= hi) continue
            val k = min(nb - 1, ((pd[i] - lo) / step).toInt())
            cnt[k]++
            val dist = max(base - po[i], 0.0)
            if (dist < mins[k]) mins[k] = dist
        }
        var sum = 0.0
        var m = 0
        for (k in 0 until nb) if (cnt[k] >= 3) { sum += min(mins[k], cap); m++ }
        return if (m >= 6) sum / m else 1e9
    }

    /** Tỉ lệ đoạn dọc cạnh có nét chữ chạm sát đường (độ "được chữ đỡ"). */
    private fun supportFrac(tp: TextPts, ln: Ln, long: Double, nb: Int = 24): Double {
        val pd = projOf(tp.xs, tp.ys, ln.d)
        val po = projOf(tp.xs, tp.ys, ln.o)
        val base = ln.p dot ln.o
        val lo = pct(pd, 2.0)
        val hi = pct(pd, 98.0)
        val step = (hi - lo) / nb
        if (step <= 0) return 0.0
        val mins = DoubleArray(nb) { Double.MAX_VALUE }
        val cnt = IntArray(nb)
        for (i in pd.indices) {
            if (pd[i] < lo || pd[i] >= hi) continue
            val k = min(nb - 1, ((pd[i] - lo) / step).toInt())
            cnt[k]++
            val dist = max(base - po[i], 0.0)
            if (dist < mins[k]) mins[k] = dist
        }
        var hit = 0
        for (k in 0 until nb) if (cnt[k] >= 2 && mins[k] < 0.014 * long) hit++
        return hit.toDouble() / nb
    }

    private fun textLines(tp: TextPts, long: Double): List<Ln>? {
        val hull = hullOf(tp)
        if (hull.size < 3) return null
        val la = try { linesA(tp, hull) } catch (e: Exception) { null }
        val lb = try { linesB(tp, hull) } catch (e: Exception) { null }
        val cands = listOfNotNull(la, lb)
        if (cands.isEmpty()) return null
        if (cands.size == 1) return cands[0]
        val fin = ArrayList<Ln>()
        for (i in 0 until 4) {
            var best = cands[0][i]
            var bs = tightness(tp, best, long)
            for (c in 1 until cands.size) {
                val s = tightness(tp, cands[c][i], long)
                if (s < bs) { bs = s; best = cands[c][i] }
            }
            fin.add(best)
        }
        val sup = DoubleArray(4) { supportFrac(tp, fin[it], long) }
        for ((i, j) in listOf(0 to 2, 2 to 0, 1 to 3, 3 to 1)) {
            val ang = Math.toDegrees(acos(clip(abs(fin[i].d dot fin[j].d), 0.0, 1.0)))
            if (sup[i] < 0.45 && sup[j] >= 0.6 && ang > 14) {
                val dj = fin[j].d
                val di = if ((dj dot fin[i].d) > 0) dj else -dj
                var oo = V(di.y, -di.x)
                if ((oo dot fin[i].o) < 0) oo = -oo
                val off = maxProj(tp, oo, 99.8)
                fin[i] = Ln(oo * off, di, oo)
            }
        }
        val q = quadFromLines(fin)
        if (q != null && isConvex(q)) return fin
        return la ?: lb
    }

    // ================================================================== 1. viền sách
    /**
     * Bắt dính các cạnh [snapIds] (0=trên,1=phải,2=dưới,3=trái) vào mép giấy thật quanh vị trí dự đoán [prior].
     * [pq] = tứ giác từ prior, [s] = px/cm, [sup] = độ được chữ đỡ của từng cạnh. Cạnh ngoài snapIds giữ nguyên.
     */
    private fun snapSides(
        small: Mat, tp: TextPts, prior: List<Ln>, pq: Array<V>, s: Double,
        sup: DoubleArray, snapIds: Set<Int>
    ): List<Ln> {
        val w = small.cols()
        val h = small.rows()
        val long = max(w, h).toDouble()
        val lab = Mat()
        Imgproc.cvtColor(small, lab, Imgproc.COLOR_BGR2Lab)
        val chs = ArrayList<Mat>()
        Core.split(lab, chs)
        val lf = Mat()
        chs[0].convertTo(lf, CvType.CV_32F)
        // ---- tín hiệu mép: độ sáng đã "đóng" (xoá chữ) + sắc độ giống giấy ----
        val kc = odd(max(9.0, 0.02 * long))
        val lc = Mat()
        Imgproc.morphologyEx(
            lf, lc, Imgproc.MORPH_CLOSE,
            Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(kc.toDouble(), kc.toDouble()))
        )
        Imgproc.GaussianBlur(lc, lc, Size(0.0, 0.0), 1.0)
        val ls = floats(lc)
        val aArr = FloatArray(w * h)
        val bArr = FloatArray(w * h)
        run {
            val ta = Mat(); val tb = Mat()
            chs[1].convertTo(ta, CvType.CV_32F, 1.0, -128.0)
            chs[2].convertTo(tb, CvType.CV_32F, 1.0, -128.0)
            ta.get(0, 0, aArr); tb.get(0, 0, bArr)
            ta.release(); tb.release()
        }
        val cx0 = (0.3 * w).toInt(); val cx1 = (0.7 * w).toInt()
        val cy0 = (0.3 * h).toInt(); val cy1 = (0.7 * h).toInt()
        val roiL = ArrayList<Double>()
        for (y in cy0 until cy1) for (x in cx0 until cx1) roiL.add(ls[y * w + x].toDouble())
        val roiArr = roiL.toDoubleArray()
        val bigP = pct(roiArr, 85.0)
        val t60 = pct(roiArr, 60.0)
        val selA = ArrayList<Double>()
        val selB = ArrayList<Double>()
        for (y in cy0 until cy1) for (x in cx0 until cx1) {
            if (ls[y * w + x] >= t60) { selA.add(aArr[y * w + x].toDouble()); selB.add(bArr[y * w + x].toDouble()) }
        }
        val ma = median(selA.toDoubleArray())
        val mbb = median(selB.toDoubleArray())
        val chromM = Mat(h, w, CvType.CV_32F)
        val chromRaw = FloatArray(w * h) {
            val d = hypot(aArr[it] - ma, bArr[it] - mbb) / 12.0
            exp(-d * d).toFloat()
        }
        chromM.put(0, 0, chromRaw)
        Imgproc.GaussianBlur(chromM, chromM, Size(0.0, 0.0), 2.0)
        val chrom = floats(chromM)

        val scales = intArrayOf(2, 4, 7, 11)
        val gam = doubleArrayOf(1.0, 0.97, 0.93, 0.88)
        val kRays = 28
        val finalL = ArrayList<Ln>()

        for (i in 0 until 4) {
            val pl = prior[i]
            if (i !in snapIds) { finalL.add(pl); continue }
            val o = pl.o
            val weak = sup[i] < 0.45
            val band = max(16.0, 2.4 * s * (if (weak) 2.4 else 1.0)).toInt()
            val a = pq[i]
            val b = pq[(i + 1) % 4]
            val n = 2 * band + 1
            val cx = DoubleArray(kRays)
            val cy = DoubleArray(kRays)
            for (k in 0 until kRays) {
                val t = 0.1 + 0.8 * k / (kRays - 1)
                cx[k] = a.x + (b.x - a.x) * t
                cy[k] = a.y + (b.y - a.y) * t
            }
            val eArr = Array(kRays) { DoubleArray(n) }
            val pcArr = Array(kRays) { DoubleArray(n) }
            val inb = Array(kRays) { BooleanArray(n) }
            val lr = DoubleArray(n)
            val cr = DoubleArray(n)
            for (k in 0 until kRays) {
                for (j in 0 until n) {
                    val u = (j - band).toDouble()
                    val px = cx[k] + u * o.x
                    val py = cy[k] + u * o.y
                    lr[j] = bil(ls, w, h, px, py)
                    cr[j] = bil(chrom, w, h, px, py)
                    inb[k][j] = px >= 0 && px <= w - 1 && py >= 0 && py <= h - 1
                }
                val cl = DoubleArray(n + 1)
                val cc = DoubleArray(n + 1)
                for (j in 0 until n) { cl[j + 1] = cl[j] + lr[j]; cc[j + 1] = cc[j] + cr[j] }
                for (si in scales.indices) {
                    val sc2 = scales[si]
                    val g = gam[si]
                    for (j in sc2 until n - sc2) {
                        val lin = (cl[j] - cl[j - sc2]) / sc2
                        val lout = (cl[j + sc2 + 1] - cl[j + 1]) / sc2
                        val dl = lin - lout
                        eArr[k][j] = max(eArr[k][j], g * (max(dl, 0.0) + 0.6 * max(-dl, 0.0)))
                        val cin = (cc[j] - cc[j - sc2]) / sc2
                        val cout = (cc[j + sc2 + 1] - cc[j + 1]) / sc2
                        pcArr[k][j] = max(pcArr[k][j], g * clip(cin - cout, 0.0, 1.0))
                    }
                }
            }
            val allE = DoubleArray(kRays * n)
            for (k in 0 until kRays) System.arraycopy(eArr[k], 0, allE, k * n, n)
            val eNorm = max(pct(allE, 97.0), 0.05 * bigP)

            val ut = pct(projOf(tp.xs, tp.ys, o), 99.7) - (pl.p dot o)
            val ptsX = DoubleArray(kRays)
            val ptsY = DoubleArray(kRays)
            val conf = DoubleArray(kRays)
            val sRow = DoubleArray(n)
            for (k in 0 until kRays) {
                for (j in 0 until n) {
                    val u = (j - band).toDouble()
                    val en = clip(eArr[k][j] / eNorm, 0.0, 1.0)
                    eArr[k][j] = en
                    val gate = clip((u - ut - 0.12 * s) / (0.1 * s), 0.05, 1.0)
                    val z = u / (0.6 * s + 1e-6) * 0.9
                    val pri = exp(-0.5 * z * z)
                    sRow[j] = if (!inb[k][j] || j < 6 || j >= n - 6) 0.0
                    else gate * (0.60 * en + 0.30 * pcArr[k][j] + 0.10 * pri)
                }
                var jb = 0
                for (j in 1 until n) if (sRow[j] > sRow[jb]) jb = j
                val smax = sRow[jb]
                var ex = -1.0
                for (j in 0 until n) if (!(j >= jb - 7 && j < jb + 8) && sRow[j] > ex) ex = sRow[j]
                val s2 = max(ex, median(sRow))
                conf[k] = clip((smax - s2) / 0.25, 0.0, 1.0) * clip(eArr[k][jb] / 0.35, 0.0, 1.0)
                val u = (jb - band).toDouble()
                ptsX[k] = cx[k] + o.x * u
                ptsY[k] = cy[k] + o.y * u
            }

            // RANSAC có trọng số độ tin cậy
            val tol = 0.005 * long
            val rnd = Random((11 + i).toLong())
            val valid = (0 until kRays).filter { conf[it] > 0.2 }
            var bestIn: BooleanArray? = null
            var bsc = 0.0
            val sl = (b - a).len()
            if (valid.size >= 5) {
                for (iter in 0 until 120) {
                    val p = valid[rnd.nextInt(valid.size)]
                    val q = valid[rnd.nextInt(valid.size)]
                    if (p == q) continue
                    var dx = ptsX[q] - ptsX[p]
                    var dy = ptsY[q] - ptsY[p]
                    val ln = hypot(dx, dy)
                    if (ln < 0.25 * sl) continue
                    dx /= ln; dy /= ln
                    val inl = BooleanArray(kRays)
                    var sc3 = 0.0
                    for (k in 0 until kRays) {
                        val dist = abs((ptsX[k] - ptsX[p]) * (-dy) + (ptsY[k] - ptsY[p]) * dx)
                        if (dist < tol && conf[k] > 0.2) { inl[k] = true; sc3 += conf[k] }
                    }
                    if (sc3 > bsc) { bsc = sc3; bestIn = inl }
                }
            }
            var line = pl
            val bi = bestIn
            val nIn = bi?.count { it } ?: 0
            if (bi != null && nIn >= 5 && bsc / nIn > 0.45 && nIn >= 0.22 * kRays) {
                var sw = 0.0; var mx = 0.0; var my = 0.0
                for (k in 0 until kRays) if (bi[k]) { sw += conf[k]; mx += conf[k] * ptsX[k]; my += conf[k] * ptsY[k] }
                mx /= sw; my /= sw
                var sxx = 0.0; var sxy = 0.0; var syy = 0.0
                for (k in 0 until kRays) if (bi[k]) {
                    val dx = ptsX[k] - mx
                    val dy = ptsY[k] - my
                    sxx += conf[k] * dx * dx; sxy += conf[k] * dx * dy; syy += conf[k] * dy * dy
                }
                val ang = 0.5 * atan2(2 * sxy, sxx - syy)
                var dv = V(cos(ang), sin(ang))
                if ((dv dot pl.d) < 0) dv = -dv
                var o3 = V(dv.y, -dv.x)
                if ((o3 dot o) < 0) o3 = -o3
                val gateDeg = if (weak) 22.0 else 8.0
                if (abs(dv dot pl.d) > cos(Math.toRadians(gateDeg))) line = Ln(V(mx, my), dv, o3)
            }
            finalL.add(line)
        }
        return finalL
    }

    /**
     * Bước 2 cho TỪNG TRANG: [crop] là trang cắt thô KÈM NỀN, [inner] = (x0,y0,x1,y1) vị trí trang dự kiến trong crop (px),
     * [ppc] = px/cm của crop, [outer] = các cạnh cần dò lại (0=trên,1=phải,2=dưới,3=trái; cạnh gáy không dò).
     * Trả về 4 góc [tl,tr,br,bl] trong toạ độ crop, hoặc null nếu không đủ chữ.
     */
    fun refinePage(crop: Mat, inner: DoubleArray, ppc: Double, outer: Set<Int>): Array<Point>? {
        val sc = 700.0 / max(crop.cols(), crop.rows())
        val small = Mat()
        Imgproc.resize(crop, small, Size(), sc, sc, Imgproc.INTER_AREA)
        val w = small.cols()
        val h = small.rows()
        val long = max(w, h).toDouble()
        val lab = Mat()
        Imgproc.cvtColor(small, lab, Imgproc.COLOR_BGR2Lab)
        val chs = ArrayList<Mat>()
        Core.split(lab, chs)
        val tp = textPoints(chs[0], w, h, long)
        if (tp.n < 100) return null
        val x0 = inner[0] * sc; val y0 = inner[1] * sc; val x1 = inner[2] * sc; val y1 = inner[3] * sc
        val prior = listOf(
            Ln(V(0.0, y0), V(1.0, 0.0), V(0.0, -1.0)),   // trên
            Ln(V(x1, 0.0), V(0.0, 1.0), V(1.0, 0.0)),    // phải
            Ln(V(0.0, y1), V(1.0, 0.0), V(0.0, 1.0)),    // dưới
            Ln(V(x0, 0.0), V(0.0, 1.0), V(-1.0, 0.0))    // trái
        )
        val pq = quadFromLines(prior) ?: return null
        val s = ppc * sc
        val sup = DoubleArray(4) { 1.0 }
        val fin = snapSides(small, tp, prior, pq, s, sup, outer)
        val safe = fin.map { ln ->
            val need = pct(projOf(tp.xs, tp.ys, ln.o), 99.9) + 0.15 * s - (ln.p dot ln.o)
            if (need > 0) Ln(ln.p + ln.o * need, ln.d, ln.o) else ln
        }
        var q = quadFromLines(safe)
        var ok = q != null
        if (ok) {
            val ar = area(q!!)
            val ap = area(pq)
            ok = isConvex(q) && ar > 0.8 * ap && ar < 1.25 * ap
        }
        if (!ok) q = pq
        val res = orderV(q!!.toList())
        return Array(4) { Point(res[it].x / sc, res[it].y / sc) }
    }

    /**
     * [spreadWcm] x [spreadHcm]: kích thước thật của sách mở (2 trang), cm.
     * Trả về [tl, tr, br, bl] theo toạ độ ảnh gốc, hoặc null nếu không tìm được (ví dụ trang không có chữ).
     */
    fun detectQuad(bgr: Mat, spreadWcm: Double, spreadHcm: Double): Array<Point>? {
        val sc = 700.0 / max(bgr.cols(), bgr.rows())
        val small = Mat()
        Imgproc.resize(bgr, small, Size(), sc, sc, Imgproc.INTER_AREA)
        val w = small.cols()
        val h = small.rows()
        val long = max(w, h).toDouble()
        val lab = Mat()
        Imgproc.cvtColor(small, lab, Imgproc.COLOR_BGR2Lab)
        val chs = ArrayList<Mat>()
        Core.split(lab, chs)
        val l8 = chs[0]
        val lf = Mat()
        l8.convertTo(lf, CvType.CV_32F)

        val tp = textPoints(l8, w, h, long)
        if (tp.n < 200) return null
        val lines = textLines(tp, long) ?: return null
        val tq = quadFromLines(lines) ?: return null

        // ---- tiên nghiệm lề theo kích thước sách ----
        val pw = spreadWcm / 2.0
        val mo = 0.095 * pw
        val mi = 0.075 * pw
        val mt = 0.060 * spreadHcm
        val mbt = 0.065 * spreadHcm
        val tw = spreadWcm - 2 * (mo + mi)
        val th = spreadHcm - (mt + mbt)
        val topw = ((tq[1] - tq[0]).len() + (tq[2] - tq[3]).len()) / 2
        val sideh = ((tq[3] - tq[0]).len() + (tq[2] - tq[1]).len()) / 2
        val s = sqrt((topw / tw) * (sideh / th))      // px/cm trên ảnh thu nhỏ
        if (!s.isFinite() || s <= 1e-3) return null
        val expM = doubleArrayOf(mt, mo, mbt, mo)     // top, right, bottom, left
        val prior = (0 until 4).map { Ln(lines[it].p + lines[it].o * (expM[it] * s), lines[it].d, lines[it].o) }
        val pq = quadFromLines(prior) ?: return null

        val sup = DoubleArray(4) { supportFrac(tp, lines[it], long) }
        val finalL = snapSides(small, tp, prior, pq, s, sup, setOf(0, 1, 2, 3))

        // an toàn: mọi nét chữ nằm trong quad (đệm 0.15 cm)
        val safe = finalL.map { ln ->
            val need = pct(projOf(tp.xs, tp.ys, ln.o), 99.9) + 0.15 * s - (ln.p dot ln.o)
            if (need > 0) Ln(ln.p + ln.o * need, ln.d, ln.o) else ln
        }
        var q = quadFromLines(safe)
        var ok = q != null
        if (ok) {
            val ar = area(q!!)
            val ap = area(pq)
            ok = isConvex(q) && ar > 0.8 * ap && ar < 1.35 * ap
        }
        if (!ok) q = pq
        val fin = q!!.map { Point(clip(it.x, 0.0, w - 1.0) / sc, clip(it.y, 0.0, h - 1.0) / sc) }
        val res = orderV(fin.map { V(it.x, it.y) })
        return Array(4) { Point(res[it].x, res[it].y) }
    }

    // ================================================================== 2. gáy sách
    /**
     * Vị trí cắt đôi trên ảnh đã nắn phẳng: bóng gáy + khoảng trống giữa hai khối chữ + gần giữa;
     * kẹp vào khoảng không có chữ và đặt gần tâm khoảng đó để không lẫn chữ trang bên kia.
     */
    fun findCut(spread: Mat): Cut {
        val sc = 700.0 / spread.cols()
        val small = Mat()
        Imgproc.resize(spread, small, Size(), sc, sc, Imgproc.INTER_AREA)
        val w = small.cols()
        val h = small.rows()
        val g = Mat()
        Imgproc.cvtColor(small, g, Imgproc.COLOR_BGR2GRAY)
        val g32 = Mat()
        g.convertTo(g32, CvType.CV_32F)
        val kc = odd(max(9.0, 0.025 * max(h, w)))
        val lc = Mat()
        Imgproc.morphologyEx(
            g32, lc, Imgproc.MORPH_CLOSE,
            Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(kc.toDouble(), kc.toDouble()))
        )
        val x0 = (0.3 * w).toInt()
        val x1 = (0.7 * w).toInt()
        val tm = textMask(g, x0, x1)
        val mx = (0.03 * w).toInt()
        val my = (0.03 * h).toInt()
        val zero = Scalar(0.0)
        tm.submat(0, my, 0, w).setTo(zero)
        tm.submat(h - my, h, 0, w).setTo(zero)
        tm.submat(0, h, 0, mx).setTo(zero)
        tm.submat(0, h, w - mx, w).setTo(zero)

        val col = Mat()
        Core.reduce(lc.submat((0.1 * h).toInt(), (0.9 * h).toInt(), 0, w), col, 0, Core.REDUCE_AVG, CvType.CV_32F)
        val kk = odd(max(5.0, w / 40.0))
        Imgproc.GaussianBlur(col, col, Size(kk.toDouble(), 1.0), 0.0)
        val cm = floats(col)
        val ref = pct(DoubleArray(((0.9 * w).toInt() - (0.1 * w).toInt())) { cm[(0.1 * w).toInt() + it].toDouble() }, 85.0)
        val dn = DoubleArray(w) { clip(clip(1 - cm[it] / ref, 0.0, 1.0) / 0.15, 0.0, 1.0) }

        val colSum = Mat()
        Core.reduce(tm, colSum, 0, Core.REDUCE_SUM, CvType.CV_32F)
        val cs = floats(colSum)
        val raw = BooleanArray(w) { cs[it] / h > 0.004 }
        val tc = BooleanArray(w) { x -> raw[x] || (x > 0 && raw[x - 1]) || (x < w - 1 && raw[x + 1]) }
        val dist = DoubleArray(w)
        var last = -1e9
        for (x in 0 until w) { if (tc[x]) last = x.toDouble(); dist[x] = x - last }
        last = 1e9
        for (x in w - 1 downTo 0) { if (tc[x]) last = x.toDouble(); dist[x] = min(dist[x], last - x) }

        var xb = w / 2
        var bv = -1e9
        for (x in x0 until x1) {
            val gap = clip(dist[x] / (0.025 * w), 0.0, 1.0)
            val z = (x - w / 2.0) / (0.08 * w)
            val s = 0.55 * dn[x] + 0.25 * gap + 0.20 * exp(-0.5 * z * z)
            if (s > bv) { bv = s; xb = x }
        }
        val margin = max(2, (0.008 * w).toInt())
        var xl = xb
        while (xl > 0 && !tc[xl]) xl--
        var xr = xb
        while (xr < w - 1 && !tc[xr]) xr++
        val risky: Boolean
        val xf: Int
        if (xr - xl >= 2 * margin + 2) {
            val ctr = (xl + xr) / 2.0
            val gp = (xr - xl).toDouble()
            val c1 = clip(xb.toDouble(), ctr - 0.25 * gp, ctr + 0.25 * gp)
            xf = clip(c1, (xl + margin).toDouble(), (xr - margin).toDouble()).toInt()
            risky = false
        } else {
            xf = xb
            risky = true
        }
        val xFull = Math.round(xf / sc).toInt().coerceIn(1, spread.cols() - 1)
        small.release(); g.release(); g32.release(); lc.release(); tm.release(); col.release(); colSum.release()
        return Cut(xFull, risky)
    }
}
