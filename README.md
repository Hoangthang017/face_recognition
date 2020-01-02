# face_recognition
Đồ án môn học CS336.K11, Trường đại học Công Nghệ Thông Tin, ĐHQG TPHCM - UIT
Nhóm gồm 3 thành viên:
- Đỗ Minh Tuấn, 16521545
- Đào Khả Phong, 16520922
- Phan Đình Nguyên, 16520850

Giảng viên hướng dẫn:
- Ths. Đỗ Văn Tiến

Hướng dẫn sử dụng:
- Vào link https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/ để xem hướng dẫn cài vggface và mtcnn
- Keras 2.2.5, Tensorflow 1.14, nếu cài phiên bản lớn hơn có thể sẽ chạy ko được.
- Cài Flask, matplotlib
- Vào link https://drive.google.com/open?id=1qmqKiZ6p-35sdxPwLHrW-_uX2_QS6tkV copy tất cả trong thư mục images paste vào thư mục images trong project, trong đó faces là ảnh của folder train trong tập data vn_celeb, uploads là folder lưu ảnh input.
- Để file embeded_face_train_resnet50_vptree_new.pickle vào thư mục gốc của project. File này lưu vector đặc trưng của các ảnh trong folder faces.
- Gõ python app.py trên terminal để khởi động ứng dụng, dùng trình duyệt vào link localhost:5000 để sử dụng ứng dụng.