package com.bookscan.app

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var spSplit: Spinner
    private lateinit var cbCrop: CheckBox
    private lateinit var cbDeskew: CheckBox
    private lateinit var cbMerge: CheckBox
    private lateinit var etSpreadW: EditText
    private lateinit var etSpreadH: EditText
    private lateinit var etMargin: EditText
    private lateinit var tvStatus: TextView
    private lateinit var tvQueue: TextView
    private lateinit var progress: ProgressBar
    private lateinit var results: LinearLayout
    private lateinit var btnCamera: Button
    private lateinit var btnPick: Button
    private lateinit var btnProcess: Button
    private lateinit var btnClear: Button
    private lateinit var btnPdf: Button

    private val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    private var counter = 0
    private val queue = mutableListOf<Uri>()      // ảnh gốc chờ xử lý
    private val saved = mutableListOf<Pair<Uri, Double>>()   // trang đã lưu + px/cm
    private var camUri: Uri? = null
    private var busy = false

    private val pickImages =
        registerForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
            if (uris.isNotEmpty()) addToQueue(uris)
        }

    // Chụp liên tục: chụp xong 1 ảnh thì mở lại camera; huỷ/quay lại để dừng.
    private val takePicture =
        registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
            if (ok) {
                camUri?.let { addToQueue(listOf(it)) }
                launchCamera()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "Không nạp được OpenCV", Toast.LENGTH_LONG).show()
        }

        spSplit = findViewById(R.id.spSplit)
        cbCrop = findViewById(R.id.cbCrop)
        cbDeskew = findViewById(R.id.cbDeskew)
        cbMerge = findViewById(R.id.cbMerge)
        etSpreadW = findViewById(R.id.etSpreadW)
        etSpreadH = findViewById(R.id.etSpreadH)
        etMargin = findViewById(R.id.etMargin)
        tvStatus = findViewById(R.id.tvStatus)
        tvQueue = findViewById(R.id.tvQueue)
        progress = findViewById(R.id.progress)
        results = findViewById(R.id.results)
        btnCamera = findViewById(R.id.btnCamera)
        btnPick = findViewById(R.id.btnPick)
        btnProcess = findViewById(R.id.btnProcess)
        btnClear = findViewById(R.id.btnClear)
        btnPdf = findViewById(R.id.btnPdf)

        spSplit.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item,
            listOf("Tự động", "Luôn tách đôi", "Không tách (1 trang)")
        )

        btnCamera.setOnClickListener { launchCamera() }
        btnPick.setOnClickListener { pickImages.launch("image/*") }
        btnProcess.setOnClickListener { processQueue() }
        btnClear.setOnClickListener { queue.clear(); updateQueueUi() }
        btnPdf.setOnClickListener { exportPdf() }
        val pref = getSharedPreferences("scan", MODE_PRIVATE)
        pref.getString("sw", null)?.let { etSpreadW.setText(it) }
        pref.getString("sh", null)?.let { etSpreadH.setText(it) }
        pref.getString("mg3", null)?.let { etMargin.setText(it) }
        updateQueueUi()
    }

    private fun options() = ScanOptions(
        split = SplitMode.values()[spSplit.selectedItemPosition],
        autoCrop = cbCrop.isChecked,
        deskew = cbDeskew.isChecked,
        merge = cbMerge.isChecked,
        spreadWcm = (etSpreadW.text.toString().replace(',', '.').toDoubleOrNull() ?: 30.0).coerceIn(10.0, 120.0),
        spreadHcm = (etSpreadH.text.toString().replace(',', '.').toDoubleOrNull() ?: 21.0).coerceIn(10.0, 80.0),
        marginCm = (etMargin.text.toString().replace(',', '.').toDoubleOrNull() ?: 0.5).coerceIn(0.0, 5.0)
    )

    private fun addToQueue(uris: List<Uri>) {
        queue.addAll(uris)
        updateQueueUi()
    }

    private fun updateQueueUi() {
        tvQueue.text = "Hàng đợi: ${queue.size} ảnh"
        btnProcess.text = if (queue.isEmpty()) "Xử lý hàng đợi" else "Xử lý ${queue.size} ảnh"
        btnProcess.isEnabled = !busy && queue.isNotEmpty()
        btnClear.isEnabled = !busy && queue.isNotEmpty()
    }

    private fun launchCamera() {
        val dir = File(cacheDir, "images").apply { mkdirs() }
        val f = File(dir, "cap_${System.currentTimeMillis()}.jpg")
        val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", f)
        camUri = uri
        takePicture.launch(uri)
    }

    private fun setBusy(b: Boolean) {
        busy = b
        progress.visibility = if (b) View.VISIBLE else View.GONE
        btnCamera.isEnabled = !b
        btnPick.isEnabled = !b
        btnPdf.isEnabled = !b
        updateQueueUi()
    }

    private fun processQueue() {
        if (queue.isEmpty()) return
        val opt = options()
        getSharedPreferences("scan", MODE_PRIVATE).edit()
            .putString("sw", etSpreadW.text.toString())
            .putString("sh", etSpreadH.text.toString())
            .putString("mg3", etMargin.text.toString()).apply()
        val work = queue.toList()
        setBusy(true)
        lifecycleScope.launch {
            var total = 0
            var errors = 0
            for ((i, uri) in work.withIndex()) {
                tvStatus.text = "Đang xử lý ảnh ${i + 1}/${work.size}…"
                try {
                    val outs = withContext(Dispatchers.Default) { processOne(uri, opt) }
                    for ((u, th) in outs) addThumb(u, th)
                    total += outs.size
                    queue.remove(uri)
                    updateQueueUi()
                } catch (e: Throwable) {
                    errors++
                    tvStatus.text = "Lỗi ở ảnh ${i + 1}: ${e.message}"
                }
            }
            tvStatus.text = "Xong: $total trang đã lưu vào Pictures/BookScanner" +
                if (errors > 0) " ($errors ảnh lỗi, còn trong hàng đợi)" else ""
            setBusy(false)
        }
    }

    private fun processOne(uri: Uri, opt: ScanOptions): List<Pair<Uri, Bitmap>> {
        val bmp = Exporter.decode(this, uri)
        val res = BookScanner.process(bmp, opt)
        bmp.recycle()
        val dpi = Math.round(res.pxPerCm * 2.54).toInt()
        return res.pages.map { p ->
            counter++
            val name = "scan_${stamp}_%03d".format(counter)
            val u = Exporter.saveJpeg(this, p, name, dpi)
            val th = Bitmap.createScaledBitmap(p, 500, (500f * p.height / p.width).toInt().coerceAtLeast(1), true)
            p.recycle()
            saved.add(u to res.pxPerCm)
            u to th
        }
    }

    private fun addThumb(uri: Uri, th: Bitmap) {
        val iv = ImageView(this).apply {
            setImageBitmap(th)
            adjustViewBounds = true
            setPadding(0, 8, 0, 8)
            setOnClickListener {
                startActivity(Intent(Intent.ACTION_VIEW).setDataAndType(uri, "image/jpeg")
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION))
            }
        }
        results.addView(iv, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
    }

    private fun exportPdf() {
        if (saved.isEmpty()) {
            Toast.makeText(this, "Chưa có trang nào", Toast.LENGTH_SHORT).show()
            return
        }
        setBusy(true)
        tvStatus.text = "Đang tạo PDF…"
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) { Exporter.savePdf(this@MainActivity, saved.toList(), "scan_$stamp") }
                tvStatus.text = "Đã lưu PDF vào Download/BookScanner"
            } catch (e: Throwable) {
                tvStatus.text = "Lỗi PDF: ${e.message}"
            }
            setBusy(false)
        }
    }
}
