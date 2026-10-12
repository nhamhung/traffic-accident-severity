"""Giới thiệu dự án bằng ngôn ngữ gần gũi với học sinh, sinh viên."""

import streamlit as st

from . import shared


def render() -> None:
    st.title("🚦 Dự báo mức độ nghiêm trọng của tai nạn giao thông")
    st.caption("Một dự án khoa học dữ liệu về an toàn giao thông đô thị")

    st.markdown(
        "Trong dự án này, em thử trả lời một câu hỏi gần gũi: **nếu biết hoàn cảnh của một vụ tai nạn, "
        "chúng ta có thể nhận ra sớm vụ nào có nguy cơ gây thương tích nặng hoặc tử vong không?**"
    )
    st.info(
        "Mục tiêu không phải là đoán chắc tương lai. Mục tiêu là tìm ra những dấu hiệu đáng chú ý để con người "
        "biết trường hợp nào cần xem xét kỹ hơn."
    )

    st.subheader("Vì sao em chọn đề tài này?")
    st.markdown(
        "Một vụ tai nạn có thể xảy ra trong nhiều hoàn cảnh khác nhau: ban ngày hay ban đêm, có đèn đường hay không, "
        "ở giao lộ nào, liên quan đến loại xe gì và người lái thuộc nhóm tuổi nào. Nhìn từng yếu tố riêng lẻ rất khó "
        "thấy bức tranh chung. Vì vậy, em dùng khoa học dữ liệu để **kết hợp nhiều dấu hiệu** và tìm ra các mẫu rủi ro."
    )

    st.subheader("Dự án sử dụng dữ liệu nào?")
    left, right = st.columns(2)
    left.markdown(
        "### 📍 Nguồn dữ liệu\n"
        "Hồ sơ tai nạn giao thông đường bộ do cảnh sát **Addis Ababa, Ethiopia** ghi nhận trong giai đoạn "
        "**2017–2020** và được công bố công khai cho mục đích học tập."
    )
    right.markdown(
        "### 🧩 Dữ liệu mô tả gì?\n"
        "Mỗi hồ sơ cho biết thời gian, ánh sáng, thời tiết, đường sá, loại va chạm, phương tiện, người lái và "
        "mức độ thương tích của vụ tai nạn."
    )
    st.markdown(
        "Dự án chỉ dùng những thông tin có thể biết **trước hoặc ngay tại lúc va chạm**. Kết quả được chia thành "
        "hai nhóm dễ hiểu: **không nghiêm trọng** (thương tích nhẹ) và **nghiêm trọng** (thương tích nặng hoặc tử vong)."
    )

    st.subheader("Vì sao dữ liệu Ethiopia vẫn có ích khi nói về Việt Nam?")
    st.markdown(
        "Dữ liệu tai nạn chi tiết ở Việt Nam chưa dễ tiếp cận cho một dự án học sinh. Bộ dữ liệu này phù hợp để làm "
        "**bài toán minh họa** vì nó có đầy đủ hoàn cảnh trước tai nạn và kết quả sau tai nạn. Addis Ababa và các "
        "thành phố Việt Nam cũng có một số câu hỏi giao thông đô thị tương tự: nhiều loại phương tiện cùng sử dụng đường, "
        "người đi bộ tương tác với xe, điều kiện chiếu sáng khác nhau và các giao lộ có mức rủi ro khác nhau."
    )
    st.warning(
        "Điều được liên hệ sang Việt Nam là **cách đặt câu hỏi và phương pháp phân tích**, không phải các con số cụ thể. "
        "Muốn kết luận cho Việt Nam, chúng ta bắt buộc phải huấn luyện và kiểm tra lại bằng dữ liệu địa phương."
    )

    st.subheader("Công cụ hoạt động như thế nào?")
    steps = st.columns(4)
    steps[0].markdown("### 1. Học\nXem các vụ tai nạn trong quá khứ và kết quả thực tế.")
    steps[1].markdown("### 2. Tìm mẫu\nNhận ra những hoàn cảnh thường đi cùng nguy cơ cao hơn.")
    steps[2].markdown("### 3. Dự đoán\nƯớc tính nguy cơ cho một tình huống mới do người dùng chọn.")
    steps[3].markdown("### 4. Giải thích\nChỉ ra yếu tố nào đang kéo nguy cơ lên hoặc xuống.")

    st.subheader("Sau bài trình bày, chúng ta sẽ hiểu gì?")
    st.markdown(
        "- Cách thử dự đoán một tình huống tai nạn và kiểm tra mô hình đúng hay sai.\n"
        "- Bốn phát hiện dễ thấy nhất về **chiếu sáng, thời điểm, nhóm tuổi người lái và số phương tiện liên quan**.\n"
        "- Cách biến một kết quả dữ liệu thành ý tưởng quản lý, đồng thời hiểu giới hạn của nó."
    )
    shared.source_note()
