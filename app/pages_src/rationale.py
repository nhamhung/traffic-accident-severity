"""Giới thiệu lý do và câu hỏi trung tâm của dự án."""

import streamlit as st

from . import shared


def render() -> None:
    st.title("💡 Vì sao có dự án này?")
    st.markdown(
        "Tai nạn giao thông không chỉ phụ thuộc vào một yếu tố. **Ánh sáng, thời điểm, loại va chạm, "
        "phương tiện và người lái** có thể cùng làm thay đổi mức độ nguy hiểm. Khi nguồn lực quản lý có hạn, "
        "câu hỏi quan trọng là: **trường hợp nào cần được chú ý trước?**"
    )

    st.subheader("Câu hỏi của dự án")
    st.info(
        "Liệu những thông tin có thể quan sát ngay trước hoặc tại thời điểm va chạm có giúp nhận diện sớm "
        "nguy cơ một vụ tai nạn trở nên nghiêm trọng hay không?"
    )

    st.subheader("Dự án muốn làm được ba việc")
    columns = st.columns(3)
    columns[0].markdown(
        "### 1. Nhận diện\n"
        "Ước tính nguy cơ nghiêm trọng từ hoàn cảnh của vụ tai nạn."
    )
    columns[1].markdown(
        "### 2. Giải thích\n"
        "Cho người xem biết những yếu tố nào đang làm nguy cơ tăng hoặc giảm."
    )
    columns[2].markdown(
        "### 3. Hỗ trợ hành động\n"
        "Biến kết quả thành câu hỏi khảo sát và ưu tiên quản lý cụ thể."
    )

    st.subheader("Thông điệp chính")
    st.success(
        "Dự đoán không thay thế quyết định của con người. Giá trị của nó là giúp **phát hiện tín hiệu đáng chú ý**, "
        "**đặt câu hỏi tốt hơn** và **ưu tiên nơi cần kiểm tra trước**."
    )
    st.markdown(
        "Ứng dụng dùng một bộ dữ liệu tai nạn đô thị được công bố công khai để minh họa cách tiếp cận. "
        "Các kết quả là một **bài trình bày giáo dục**; muốn áp dụng tại Việt Nam cần kiểm chứng lại bằng dữ liệu địa phương."
    )
    shared.source_note()
