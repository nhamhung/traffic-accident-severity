"""Chuyển các phát hiện thành những bước quản lý giao thông dễ trình bày."""

import streamlit as st

from . import shared


def render() -> None:
    st.title("🚦 Ứng dụng tiềm năng vào quản lý giao thông")
    st.markdown(
        "Đến đây, câu hỏi của em không còn là “mô hình dự đoán được bao nhiêu phần trăm?”, mà là: "
        "**nếu thấy một tín hiệu rủi ro, chúng ta có thể làm gì với nó?**"
    )

    st.subheader("1. Từ ba phát hiện đến ba ý tưởng hành động")
    columns = st.columns(3)
    columns[0].markdown(
        "### 💡 Chiếu sáng\n"
        "Khảo sát các đoạn đường tối, kiểm tra đèn và khả năng quan sát trước khi quyết định nơi cần ưu tiên."
    )
    columns[1].markdown(
        "### 🌙 Thời điểm\n"
        "Cân nhắc cảnh báo, tuần tra và khả năng phản ứng vào những khung giờ cần chú ý hơn."
    )
    columns[2].markdown(
        "### 👥 Nhóm người lái\n"
        "Tìm hiểu nhu cầu khác nhau của người lái trẻ và lớn tuổi để thiết kế hướng dẫn phù hợp hơn."
    )

    st.subheader("2. Hãy tưởng tượng một tình huống gần gũi")
    st.markdown(
        "Giả sử quanh trường có một đoạn đường tối, nhiều xe đi qua vào buổi tối và công cụ liên tục đánh dấu các "
        "tình huống ở đó là đáng chú ý. Kết quả này **không đủ để kết luận** đoạn đường nguy hiểm vì thiếu đèn. "
        "Nhưng nó là lý do hợp lý để nhà trường hoặc địa phương kiểm tra ánh sáng, tốc độ xe, vạch qua đường và tầm nhìn."
    )

    st.subheader("3. Quy trình hợp lý sẽ gồm năm bước")
    st.markdown(
        "1. **Ghi nhận:** thu thập hoàn cảnh tai nạn theo cùng một cách.\n"
        "2. **Sàng lọc:** dùng công cụ để tìm tình huống cần chú ý.\n"
        "3. **Kiểm tra:** đến hiện trường và hỏi người có chuyên môn.\n"
        "4. **Thử giải pháp nhỏ:** ví dụ cải thiện chiếu sáng hoặc biển báo tại một vài điểm.\n"
        "5. **So sánh trước–sau:** chỉ mở rộng nếu giải pháp thực sự mang lại kết quả tốt."
    )

    st.subheader("4. Công cụ làm được gì — và không làm được gì?")
    can_do, cannot_do = st.columns(2)
    can_do.markdown(
        "### ✅ Có thể hỗ trợ\n"
        "- So sánh nhiều tình huống theo cùng một cách.\n"
        "- Gợi ý nơi hoặc thời điểm cần khảo sát trước.\n"
        "- Giúp giải thích vì sao một cảnh báo xuất hiện."
    )
    cannot_do.markdown(
        "### ❌ Không thể thay thế\n"
        "- Dữ liệu giao thông thực tế tại Việt Nam.\n"
        "- Kiểm tra hiện trường và kiến thức chuyên gia.\n"
        "- Trách nhiệm ra quyết định của con người."
    )

    st.subheader("5. Nếu tiếp tục dự án tại Việt Nam")
    st.markdown(
        "Bước tiếp theo hợp lý nhất là xin một bộ dữ liệu địa phương, huấn luyện lại mô hình, rồi thử nghiệm ở quy mô "
        "nhỏ. Em cũng cần trao đổi với chuyên gia giao thông để kiểm tra xem các phát hiện có hợp lý trong bối cảnh Việt Nam hay không."
    )
    st.success(
        "**Điều em rút ra từ dự án:** khoa học dữ liệu hữu ích nhất khi nó giúp con người đặt câu hỏi tốt hơn, "
        "chứ không phải khi nó cố thay con người đưa ra câu trả lời cuối cùng."
    )
    shared.source_note()
