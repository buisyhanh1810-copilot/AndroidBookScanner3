# Book Scanner (Android)

Chụp liên tục/chọn nhiều ảnh vào hàng đợi, bấm *Xử lý* sau. Mỗi ảnh là một trang sách mở (2 trang liền nhau) → app tự tìm viền sách, nắn phẳng,
tìm rãnh gáy, tách thành 2 trang theo thứ tự (trái → phải), xoá bóng, chuyển đen trắng,
chỉnh nghiêng, cắt theo nội dung + lề dư 1 cm (chỉnh được), lưu JPEG vào `Pictures/BookScanner`. Có nút gộp PDF (`Download/BookScanner`).

## Build bằng GitHub Actions
1. Tạo repo mới trên GitHub, đẩy toàn bộ thư mục này lên nhánh `main`.
2. Vào tab **Actions** → workflow **Build APK** chạy tự động (hoặc bấm *Run workflow*).
3. Chạy xong, mở lần chạy đó → mục **Artifacts** → tải `BookScanner-debug-apk`, giải nén lấy `app-debug.apk` cài lên máy.
4. Muốn có bản Release: `git tag v1.0 && git push --tags`, APK sẽ được đính vào Releases.

Không cần Gradle wrapper: workflow dùng `gradle/actions/setup-gradle` với Gradle 8.7.
Build cục bộ: Android Studio mở thư mục này, hoặc `gradle assembleDebug`.

## Ghi chú
- minSdk 29 (Android 10+), dùng OpenCV 4.9.0 từ Maven Central.
- Sách xoay dọc trong ảnh: chọn "Luôn tách đôi" (app tự xoay 90°).
- Nền quanh sách có vật sáng dính vào sách làm sai viền: bỏ tick "Tự phát hiện viền" rồi chụp sát sách.

## Thuật toán tách trang v2 (dùng kích thước thật của sách)

Nhập **rộng × cao của sách mở (cm)**, chụp bằng camera **ngang**, sách nằm ngang (trang trái – gáy – trang phải).

1. Tìm khối chữ thân văn bản → 4 đường biên khối chữ (bao lồi).
2. Từ kích thước sách suy ra px/cm và vị trí dự đoán mép giấy (biên chữ + lề điển hình).
3. Mỗi cạnh "bắt dính" vào mép giấy thật (bước nhảy sáng/tối nhiều tỉ lệ + sắc độ giấy so với nền, RANSAC); cạnh không đủ tin cậy giữ vị trí dự đoán; mọi nét chữ luôn nằm trong khung.
4. Nắn tạm cả sách để dò gáy (bóng gáy + khoảng trống chữ, dò riêng nửa trên/nửa dưới), chiếu đường gáy về ảnh gốc, rồi tách 2 bước: **(1) cắt thô từng trang kèm nền** (tứ giác mép ngoài + đường gáy, nới 1,5 cm ra ngoài); **(2) từng trang tự dò lại 3 mép ngoài** (trên/dưới/ngoài) rồi nắn bằng **homography riêng** về chữ nhật đúng 1/2 bề rộng × chiều cao nhập vào. Sau đó mới xử lý riêng từng trang (xoá bóng, đen trắng, chỉnh nghiêng, cắt theo nội dung).
5. Ảnh JPEG ghi DPI thật; PDF có đúng kích thước cm.

Không thấy chữ (trang trống) → tự dùng thuật toán cũ làm dự phòng.

**Kiểm định**: `tools/prototype` (Python/OpenCV) sinh ảnh tổng hợp có đáp án (camera nghiêng/xoay, nền gỗ/vải/tối/sáng, trang cong, gáy mờ, trang trống) và đo sai số góc (mm), chữ bị cắt, chữ lấn sang trang bên kia: `python3 run2.py 90 777`, `python3 stress.py`.
Kết quả: sai số góc trung bình ≈ 6 mm (thuật toán cũ ≈ 180 mm), 0 ca cắt vào chữ; 2/90 ca nghiêng cực đoan trên nền vải/gỗ còn lẹm nhẹ sang trang bên kia.

**Ghép lại:** tuỳ chọn "Xử lý riêng từng trang rồi ghép lại thành 1 ảnh 2 trang" – mỗi trang được xử lý độc lập (xoá bóng, đen trắng, chỉnh nghiêng, cắt nội dung) rồi ghép cạnh nhau cùng tỉ lệ px/cm.

## GitHub Actions
Mỗi lần push lên main/master, workflow `Build APK` tự build APK, rồi cập nhật Releases → **Bản mới nhất** (tải APK ở đó, không cần vào Actions). Push tag `v1.0` để tạo bản phát hành chính thức. Cũng có thể chạy tay: tab Actions → Run workflow.
