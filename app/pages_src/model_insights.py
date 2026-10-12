"""Chuyển các phát hiện thành những bước quản lý giao thông dễ trình bày."""

import streamlit as st

from . import shared


def render() -> None:
    st.title("🚦 Ứng dụng tiềm năng vào quản lý giao thông")
    st.markdown(
        "Dự án không nhằm tự động ra quyết định. Giá trị thực tế nằm ở việc giúp cơ quan quản lý "
        "**nhìn thấy tín hiệu rủi ro sớm hơn và ưu tiên nguồn lực hợp lý hơn**."
    )

    st.subheader("1. Ba hướng hành động từ những phát hiện chính")
    columns = st.columns(3)
    columns[0].markdown(
        "### 💡 Chiếu sáng\n"
        "Lập bản đồ các đoạn đường tối, kiểm tra hệ thống đèn và ưu tiên điểm có nhiều tình huống nguy cơ cao."
    )
    columns[1].markdown(
        "### 🌙 Thời điểm\n"
        "Tăng cảnh báo, tuần tra hoặc khả năng phản ứng vào những khung giờ ban đêm và rạng sáng."
    )
    columns[2].markdown(
        "### 👥 Nhóm người lái\n"
        "Thiết kế truyền thông và hỗ trợ phù hợp hơn cho người lái rất trẻ và người lái lớn tuổi."
    )

    st.subheader("2. Một quy trình hỗ trợ quyết định")
    st.markdown(
        "1. **Ghi nhận hoàn cảnh:** ánh sáng, thời điểm, đường sá, phương tiện và kiểu va chạm.\n"
        "2. **Sàng lọc nguy cơ:** dùng công cụ dự đoán để nhận diện trường hợp đáng chú ý.\n"
        "3. **Kiểm tra thực địa:** cảnh sát và kỹ sư giao thông xác minh nguyên nhân có thể can thiệp.\n"
        "4. **Chọn biện pháp:** cải thiện chiếu sáng, tổ chức giao lộ, biển báo, tốc độ hoặc truyền thông.\n"
        "5. **Đo lại kết quả:** so sánh trước và sau để biết biện pháp có thực sự hiệu quả hay không."
    )

    st.subheader("3. Ví dụ: từ cảnh báo đến hành động")
    st.markdown(
        "Nếu nhiều tình huống nguy cơ cao cùng xuất hiện tại các đoạn đường tối vào ban đêm, cơ quan quản lý có thể:\n\n"
        "- xác định các vị trí cần khảo sát trước;\n"
        "- kiểm tra độ sáng, tốc độ xe và khả năng quan sát;\n"
        "- thử cải thiện một số vị trí ưu tiên;\n"
        "- theo dõi xem số vụ nghiêm trọng có giảm sau can thiệp hay không."
    )

    st.subheader("4. Điều kiện để áp dụng tại Việt Nam")
    st.markdown(
        "- Kiểm chứng lại bằng dữ liệu giao thông tại địa phương.\n"
        "- Thử nghiệm ở quy mô nhỏ trước khi mở rộng.\n"
        "- Luôn có chuyên gia xem xét kết quả và chịu trách nhiệm quyết định.\n"
        "- Bảo vệ thông tin cá nhân và theo dõi tác động ngoài mong muốn."
    )
    st.warning(
        "Đây là dự án giáo dục. Kết quả minh họa một cách tư duy và xây dựng công cụ hỗ trợ; "
        "chưa đủ để đưa ra quyết định quản lý thực tế tại Việt Nam."
    )
    shared.source_note()
