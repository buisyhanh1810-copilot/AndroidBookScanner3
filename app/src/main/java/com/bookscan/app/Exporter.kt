package com.bookscan.app

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.pdf.PdfDocument
import android.net.Uri
import android.provider.MediaStore
import androidx.exifinterface.media.ExifInterface
import kotlin.math.max
import kotlin.math.roundToInt

object Exporter {

    fun decode(ctx: Context, uri: Uri, maxSide: Int = 3200): Bitmap {
        val cr = ctx.contentResolver
        val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        cr.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, o) }
        var sample = 1
        while (max(o.outWidth, o.outHeight) / (sample * 2) >= maxSide) sample *= 2
        val o2 = BitmapFactory.Options().apply { inSampleSize = sample }
        var bmp = cr.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, o2) }
            ?: error("Không đọc được ảnh")
        val orient = try {
            cr.openInputStream(uri)!!.use {
                ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            }
        } catch (e: Exception) { ExifInterface.ORIENTATION_NORMAL }
        val deg = when (orient) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        if (deg != 0f) {
            val m = Matrix().apply { postRotate(deg) }
            val r = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            bmp.recycle()
            bmp = r
        }
        return bmp
    }

    /** Ghi mật độ điểm ảnh (DPI) vào đầu JFIF để ảnh mở ra đúng kích thước cm thật. */
    private fun setJfifDpi(b: ByteArray, dpi: Int) {
        if (b.size > 18 && b[2] == 0xFF.toByte() && b[3] == 0xE0.toByte() &&
            b[6] == 'J'.code.toByte() && b[7] == 'F'.code.toByte() && b[8] == 'I'.code.toByte() && b[9] == 'F'.code.toByte()
        ) {
            val d = dpi.coerceIn(1, 65535)
            b[13] = 1                                   // đơn vị: dots per inch
            b[14] = (d shr 8).toByte(); b[15] = d.toByte()
            b[16] = (d shr 8).toByte(); b[17] = d.toByte()
        }
    }

    fun saveJpeg(ctx: Context, bmp: Bitmap, name: String, dpi: Int = 0, quality: Int = 92): Uri {
        val cr = ctx.contentResolver
        val v = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "$name.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/BookScanner")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = cr.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, v) ?: error("Không tạo được file")
        val bos = java.io.ByteArrayOutputStream()
        bmp.compress(Bitmap.CompressFormat.JPEG, quality, bos)
        val bytes = bos.toByteArray()
        if (dpi > 0) setJfifDpi(bytes, dpi)
        cr.openOutputStream(uri)!!.use { it.write(bytes) }
        v.clear()
        v.put(MediaStore.Images.Media.IS_PENDING, 0)
        cr.update(uri, v, null, null)
        return uri
    }

    /** [pages]: (ảnh, px/cm). Trang PDF có đúng kích thước cm thật (1 pt = 1/72 inch). */
    fun savePdf(ctx: Context, pages: List<Pair<Uri, Double>>, name: String): Uri {
        val doc = PdfDocument()
        val paint = Paint(Paint.FILTER_BITMAP_FLAG)
        pages.forEachIndexed { i, (u, ppc) ->
            val bmp = decode(ctx, u, 4000)
            val bo = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            ctx.contentResolver.openInputStream(u)!!.use { BitmapFactory.decodeStream(it, null, bo) }
            val origW = bo.outWidth; val origH = bo.outHeight     // kích thước gốc (không sample)
            val ptPerPx = if (ppc > 0) 72.0 / 2.54 / ppc else 595.0 / origW
            val pw = (origW * ptPerPx).roundToInt().coerceAtLeast(50)
            val ph = (origH * ptPerPx).roundToInt().coerceAtLeast(50)
            val page = doc.startPage(PdfDocument.PageInfo.Builder(pw, ph, i + 1).create())
            page.canvas.drawBitmap(bmp, null, Rect(0, 0, pw, ph), paint)
            doc.finishPage(page)
            bmp.recycle()
        }
        val cr = ctx.contentResolver
        val v = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, "$name.pdf")
            put(MediaStore.Downloads.MIME_TYPE, "application/pdf")
            put(MediaStore.Downloads.RELATIVE_PATH, "Download/BookScanner")
            put(MediaStore.Downloads.IS_PENDING, 1)
        }
        val uri = cr.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, v) ?: error("Không tạo được PDF")
        cr.openOutputStream(uri)!!.use { doc.writeTo(it) }
        doc.close()
        v.clear()
        v.put(MediaStore.Downloads.IS_PENDING, 0)
        cr.update(uri, v, null, null)
        return uri
    }
}
