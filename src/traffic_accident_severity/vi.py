"""Vietnamese labels for the app and the report.

The data and the model keep the dataset's original English values (so the
trained pipeline and the raw CSV stay untouched); only what people *see* is
translated, through `col()`, `val()` and `cls()`. Wording is kept simple
enough for secondary-school students.

A few raw values are garbled in the source dataset itself (e.g.
"statioNot a Pedestrianry" is "stationary" mangled by a find-and-replace in
the original release); the translations below say what they mean.
"""

CLASS = {"Severe": "Nghiêm trọng", "Not Severe": "Không nghiêm trọng"}
RAW_SEVERITY = {"Slight Injury": "Bị thương nhẹ", "Serious Injury": "Bị thương nặng", "Fatal injury": "Tử vong"}

COLUMN = {
    "Time": "Giờ xảy ra",
    "hour": "Giờ trong ngày",
    "Day_of_week": "Thứ trong tuần",
    "Age_band_of_driver": "Tuổi người lái",
    "Sex_of_driver": "Giới tính người lái",
    "Educational_level": "Trình độ học vấn người lái",
    "Vehicle_driver_relation": "Quan hệ người lái với xe",
    "Driving_experience": "Kinh nghiệm lái xe",
    "Type_of_vehicle": "Loại xe",
    "Owner_of_vehicle": "Chủ sở hữu xe",
    "Service_year_of_vehicle": "Số năm xe đã sử dụng",
    "Defect_of_vehicle": "Lỗi kỹ thuật của xe",
    "Area_accident_occured": "Khu vực xảy ra tai nạn",
    "Lanes_or_Medians": "Làn đường / dải phân cách",
    "Road_allignment": "Hình dạng con đường",
    "Types_of_Junction": "Loại giao lộ",
    "Road_surface_type": "Loại mặt đường",
    "Road_surface_conditions": "Tình trạng mặt đường",
    "Light_conditions": "Ánh sáng",
    "Weather_conditions": "Thời tiết",
    "Type_of_collision": "Kiểu va chạm",
    "Number_of_vehicles_involved": "Số xe liên quan",
    "Vehicle_movement": "Xe đang làm gì",
    "Pedestrian_movement": "Người đi bộ đang làm gì",
    "Cause_of_accident": "Nguyên nhân tai nạn",
    "Accident_severity": "Mức độ nghiêm trọng",
}

_UNKNOWN = "Không rõ"
_OTHER = "Khác"

VALUE = {
    "Day_of_week": {"Monday": "Thứ Hai", "Tuesday": "Thứ Ba", "Wednesday": "Thứ Tư", "Thursday": "Thứ Năm",
                    "Friday": "Thứ Sáu", "Saturday": "Thứ Bảy", "Sunday": "Chủ Nhật"},
    "Age_band_of_driver": {"Under 18": "Dưới 18 tuổi", "18-30": "18–30 tuổi", "31-50": "31–50 tuổi",
                           "Over 51": "Trên 51 tuổi", "Unknown": _UNKNOWN},
    "Sex_of_driver": {"Male": "Nam", "Female": "Nữ", "Unknown": _UNKNOWN},
    "Educational_level": {"Illiterate": "Không biết chữ", "Writing & reading": "Biết đọc, biết viết",
                          "Elementary school": "Tiểu học", "Junior high school": "Trung học cơ sở",
                          "High school": "Trung học phổ thông", "Above high school": "Trên THPT (cao đẳng, đại học)",
                          "Unknown": _UNKNOWN},
    "Vehicle_driver_relation": {"Owner": "Chủ xe tự lái", "Employee": "Người được thuê lái", "Other": _OTHER,
                                "Unknown": _UNKNOWN},
    "Driving_experience": {"No Licence": "Không có bằng lái", "Below 1yr": "Dưới 1 năm", "1-2yr": "1–2 năm",
                           "2-5yr": "2–5 năm", "5-10yr": "5–10 năm", "Above 10yr": "Trên 10 năm",
                           "unknown": _UNKNOWN},
    "Type_of_vehicle": {"Automobile": "Ô tô con", "Bajaj": "Xe ba bánh (Bajaj)", "Bicycle": "Xe đạp",
                        "Long lorry": "Xe tải dài (container)", "Lorry (11?40Q)": "Xe tải vừa (1,1–4 tấn)",
                        "Lorry (41?100Q)": "Xe tải lớn (4,1–10 tấn)", "Motorcycle": "Xe máy", "Other": _OTHER,
                        "Pick up upto 10Q": "Xe bán tải (đến 1 tấn)", "Public (12 seats)": "Xe khách nhỏ (12 chỗ)",
                        "Public (13?45 seats)": "Xe khách vừa (13–45 chỗ)", "Public (> 45 seats)": "Xe buýt lớn (trên 45 chỗ)",
                        "Ridden horse": "Ngựa (có người cưỡi)", "Special vehicle": "Xe chuyên dụng",
                        "Stationwagen": "Ô tô gia đình (station wagon)", "Taxi": "Taxi", "Turbo": "Xe tải nhẹ (Turbo)"},
    "Owner_of_vehicle": {"Owner": "Cá nhân", "Governmental": "Nhà nước", "Organization": "Tổ chức, công ty",
                         "Other": _OTHER},
    "Service_year_of_vehicle": {"Below 1yr": "Dưới 1 năm", "1-2yr": "1–2 năm", "2-5yrs": "2–5 năm",
                                "5-10yrs": "5–10 năm", "Above 10yr": "Trên 10 năm", "Unknown": _UNKNOWN},
    "Defect_of_vehicle": {"No defect": "Không có lỗi", "5": "Có lỗi (mã 5)", "7": "Có lỗi (mã 7)"},
    "Area_accident_occured": {"  Market areas": "Khu chợ", "  Recreational areas": "Khu vui chơi giải trí",
                              " Church areas": "Khu nhà thờ", " Hospital areas": "Khu bệnh viện",
                              " Industrial areas": "Khu công nghiệp", " Outside rural areas": "Ngoại ô",
                              "Office areas": "Khu văn phòng", "Other": _OTHER,
                              "Recreational areas": "Khu vui chơi giải trí", "Residential areas": "Khu dân cư",
                              "Rural village areas": "Làng quê", "Rural village areasOffice areas": "Làng quê / văn phòng",
                              "School areas": "Khu trường học", "Unknown": _UNKNOWN},
    "Lanes_or_Medians": {"Double carriageway (median)": "Đường đôi (có dải phân cách giữa)", "One way": "Đường một chiều",
                         "Two-way (divided with broken lines road marking)": "Hai chiều, vạch kẻ đứt",
                         "Two-way (divided with solid lines road marking)": "Hai chiều, vạch kẻ liền",
                         "Undivided Two way": "Hai chiều, không phân cách", "Unknown": _UNKNOWN, "other": _OTHER},
    "Road_allignment": {"Escarpments": "Đường sát vách dốc", "Gentle horizontal curve": "Khúc cua nhẹ",
                        "Sharp reverse curve": "Khúc cua gấp liên tiếp",
                        "Steep grade downward with mountainous terrain": "Dốc xuống đường núi",
                        "Steep grade upward with mountainous terrain": "Dốc lên đường núi",
                        "Tangent road with flat terrain": "Đường thẳng, bằng phẳng",
                        "Tangent road with mild grade and flat terrain": "Đường thẳng, hơi dốc",
                        "Tangent road with mountainous terrain and": "Đường thẳng, vùng núi",
                        "Tangent road with rolling terrain": "Đường thẳng, đồi thoải"},
    "Types_of_Junction": {"Crossing": "Ngã tư có vạch qua đường", "No junction": "Không có giao lộ",
                          "O Shape": "Vòng xuyến", "Other": _OTHER, "T Shape": "Ngã ba chữ T", "Unknown": _UNKNOWN,
                          "X Shape": "Ngã tư chữ X", "Y Shape": "Ngã ba chữ Y"},
    "Road_surface_type": {"Asphalt roads": "Đường nhựa", "Asphalt roads with some distress": "Đường nhựa hư hỏng",
                          "Earth roads": "Đường đất", "Gravel roads": "Đường sỏi đá", "Other": _OTHER},
    "Road_surface_conditions": {"Dry": "Khô ráo", "Flood over 3cm. deep": "Ngập nước trên 3 cm", "Snow": "Có tuyết",
                                "Wet or damp": "Ướt, ẩm"},
    "Light_conditions": {"Daylight": "Ban ngày", "Darkness - lights lit": "Ban đêm, có đèn đường",
                         "Darkness - lights unlit": "Ban đêm, đèn đường không bật",
                         "Darkness - no lighting": "Ban đêm, không có đèn đường"},
    "Weather_conditions": {"Normal": "Bình thường", "Cloudy": "Nhiều mây", "Fog or mist": "Sương mù",
                           "Raining": "Mưa", "Raining and Windy": "Mưa và gió", "Snow": "Tuyết", "Windy": "Gió",
                           "Other": _OTHER, "Unknown": _UNKNOWN},
    "Type_of_collision": {"Collision with animals": "Đâm vào động vật", "Collision with pedestrians": "Đâm vào người đi bộ",
                          "Collision with roadside objects": "Đâm vào vật bên đường",
                          "Collision with roadside-parked vehicles": "Đâm vào xe đỗ bên đường",
                          "Fall from vehicles": "Ngã từ trên xe", "Other": _OTHER, "Rollover": "Lật xe",
                          "Unknown": _UNKNOWN, "Vehicle with vehicle collision": "Hai xe đâm nhau",
                          "With Train": "Va chạm với tàu hỏa"},
    "Vehicle_movement": {"Entering a junction": "Đang vào giao lộ", "Getting off": "Đang xuống xe",
                         "Going straight": "Đang đi thẳng", "Moving Backward": "Đang lùi", "Other": _OTHER,
                         "Overtaking": "Đang vượt xe khác", "Parked": "Đang đỗ", "Reversing": "Đang lùi xe",
                         "Stopping": "Đang dừng lại", "Turnover": "Bị lật", "U-Turn": "Đang quay đầu",
                         "Unknown": _UNKNOWN, "Waiting to go": "Đang chờ đi"},
    "Pedestrian_movement": {
        "Not a Pedestrian": "Không có người đi bộ",
        "Crossing from driver's nearside": "Băng qua đường từ phía gần người lái",
        "Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle":
            "Băng qua từ phía gần, bị xe đỗ che khuất",
        "Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle":
            "Băng qua từ phía xa, bị xe đỗ che khuất",
        "In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)":
            "Đứng hoặc chơi trên lòng đường",
        "In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle":
            "Đứng trên lòng đường, bị xe đỗ che khuất",
        "Unknown or other": "Không rõ / khác",
        "Walking along in carriageway, back to traffic": "Đi dọc lòng đường, quay lưng với xe",
        "Walking along in carriageway, facing traffic": "Đi dọc lòng đường, đối mặt với xe",
    },
    "Cause_of_accident": {"Changing lane to the left": "Chuyển làn sang trái", "Changing lane to the right": "Chuyển làn sang phải",
                          "Driving at high speed": "Chạy tốc độ cao", "Driving carelessly": "Lái xe bất cẩn",
                          "Driving to the left": "Lấn sang bên trái đường",
                          "Driving under the influence of drugs": "Lái xe khi dùng chất kích thích",
                          "Drunk driving": "Lái xe khi say rượu", "Getting off the vehicle improperly": "Xuống xe không đúng cách",
                          "Improper parking": "Đỗ xe sai quy định", "Moving Backward": "Lùi xe", "No distancing": "Không giữ khoảng cách",
                          "No priority to pedestrian": "Không nhường người đi bộ", "No priority to vehicle": "Không nhường xe khác",
                          "Other": _OTHER, "Overloading": "Chở quá tải", "Overspeed": "Vượt quá tốc độ cho phép",
                          "Overtaking": "Vượt xe ẩu", "Overturning": "Lật xe", "Turnover": "Lật xe", "Unknown": _UNKNOWN},
}


def col(name: str) -> str:
    """Vietnamese label for a column (falls back to the raw name)."""
    return COLUMN.get(name, name)


def val(column: str, value) -> str:
    """Vietnamese label for one raw value of `column` (falls back to the raw value)."""
    if value is None or (isinstance(value, float) and value != value):
        return "Không ghi nhận"
    return VALUE.get(column, {}).get(str(value), str(value))


def cls(name: str) -> str:
    """Vietnamese name of a model class or raw severity."""
    return CLASS.get(name, RAW_SEVERITY.get(name, name))
