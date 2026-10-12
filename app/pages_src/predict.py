"""Trang Dự đoán: chọn hoàn cảnh của một vụ tai nạn, xem mô hình đánh giá
khả năng nghiêm trọng và lý do.

Only pre-crash conditions are offered (driver, vehicle, road, weather,
collision) - never casualty outcomes, which the model never sees.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from . import shared
from traffic_accident_severity import config, vi

MAIN_FIELDS = ["Light_conditions", "Type_of_collision", "Number_of_vehicles_involved", "Cause_of_accident",
               "Age_band_of_driver", "Types_of_Junction"]
GROUPS = {
    "🧑 Người lái": ["Sex_of_driver", "Driving_experience", "Educational_level", "Vehicle_driver_relation"],
    "🚗 Phương tiện": ["Type_of_vehicle", "Owner_of_vehicle", "Service_year_of_vehicle", "Defect_of_vehicle",
                      "Vehicle_movement"],
    "🛣️ Con đường": ["Area_accident_occured", "Lanes_or_Medians", "Road_allignment", "Road_surface_type",
                    "Road_surface_conditions", "Weather_conditions"],
    "🚶 Người đi bộ và thời gian": ["Pedestrian_movement", "Day_of_week"],
}


def _key(col: str) -> str:
    return f"field_{col}"


def _set_from_row(row: pd.Series | dict) -> None:
    for col in config.RAW_FEATURE_COLS:
        if col == config.TIME_COL:
            continue
        value = row[col]
        if col == "Number_of_vehicles_involved":
            st.session_state[_key(col)] = int(value)
        else:
            st.session_state[_key(col)] = value if isinstance(value, str) else None
    hour = pd.to_datetime(row[config.TIME_COL], format="%H:%M:%S", errors="coerce")
    st.session_state["hour_pick"] = int(hour.hour) if not pd.isna(hour) else 12


def _store_loaded_accident(row: pd.Series) -> None:
    _set_from_row(row)
    st.session_state["loaded"] = {c: st.session_state.get(_key(c)) for c in config.RAW_FEATURE_COLS
                                  if c != config.TIME_COL} | {"hour": st.session_state["hour_pick"]}
    st.session_state["loaded_severity"] = row[config.TARGET_COL]
    st.session_state.pop("load_error", None)


def _load_random_non_severe_accident() -> None:
    accidents = shared.get_accidents_df()
    non_severe_accidents = accidents[~accidents[config.TARGET_COL].isin(config.SEVERE_CLASSES)]
    if non_severe_accidents.empty:
        st.session_state["load_error"] = "Không tìm thấy vụ tai nạn không nghiêm trọng trong dữ liệu."
        return
    _store_loaded_accident(non_severe_accidents.sample(1).iloc[0])


def _load_random_severe_accident() -> None:
    accidents = shared.get_accidents_df()
    severe_accidents = accidents[accidents[config.TARGET_COL].isin(config.SEVERE_CLASSES)]
    if severe_accidents.empty:
        st.session_state["load_error"] = "Không tìm thấy vụ tai nạn nghiêm trọng trong dữ liệu."
        return
    _store_loaded_accident(severe_accidents.sample(1).iloc[0])


def _reset() -> None:
    _set_from_row(shared.get_default_row())
    st.session_state.pop("loaded", None)
    st.session_state.pop("loaded_severity", None)
    st.session_state.pop("load_error", None)


def _current_row() -> dict:
    row = {c: st.session_state.get(_key(c)) for c in config.RAW_FEATURE_COLS if c != config.TIME_COL}
    row[config.TIME_COL] = f"{st.session_state['hour_pick']:02d}:00:00"
    return row


def _unchanged_since_load(row: dict) -> bool:
    loaded = st.session_state.get("loaded")
    if not loaded:
        return False
    return all(loaded[c] == row[c] for c in loaded if c != "hour") and loaded["hour"] == st.session_state["hour_pick"]


def _matches_loaded_accident(current: dict, reference: dict | None = None) -> bool:
    """Only a completely unchanged historical row has a valid recorded outcome."""
    if reference is None:
        reference = st.session_state.get("loaded_accident_features")
    if not reference or set(reference) != set(config.RAW_FEATURE_COLS):
        return False
    for col in config.RAW_FEATURE_COLS:
        a, b = current[col], reference[col]
        if pd.isna(a) and pd.isna(b):
            continue
        if a != b:
            return False
    return True


def _select(col: str) -> None:
    values = shared.options(col)
    if st.session_state.get(_key(col)) not in values:
        st.session_state[_key(col)] = values[0]
    st.selectbox(vi.col(col), values, key=_key(col), format_func=lambda v: vi.val(col, v))


def _explanation_chart(contrib: pd.Series, row: dict):
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(7)[::-1]

    def label(feature: str) -> str:
        if feature == "hour":
            return f"Giờ: {st.session_state['hour_pick']}h"
        value = row.get(feature)
        shown = value if feature == "Number_of_vehicles_involved" else vi.val(feature, value)
        return f"{vi.col(feature)}: {shown}"

    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(top) + 0.8))
    ax.barh([label(f) for f in top.index], top.values,
            color=[shared.SEVERE_COLOUR if v > 0 else shared.SAFE_COLOUR for v in top.values])
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xticks([])  # the units (log-odds) mean nothing to students; bar length is the message
    ax.set_xlabel("← ít nguy hiểm hơn            nguy hiểm hơn →")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    return fig


def render() -> None:
    st.title("🎯 Thử dự đoán một tình huống tai nạn")
    st.markdown(
        "Hãy chọn **hoàn cảnh** của một vụ tai nạn — ánh sáng, kiểu va chạm, số xe và người lái. "
        "Công cụ sẽ ước tính **nguy cơ vụ đó nghiêm trọng** và chỉ ra những yếu tố đáng chú ý nhất."
    )
    st.caption("Công cụ chỉ xem xét thông tin có thể quan sát trước hoặc ngay tại thời điểm va chạm.")

    try:
        severity_model = shared.get_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    if _key("Light_conditions") not in st.session_state:
        _reset()

    b1, b2 = st.columns(2)
    b1.button("✅ Lấy một vụ tai nạn không nghiêm trọng", on_click=_load_random_non_severe_accident, width="stretch")
    b2.button("⚠️ Lấy một vụ tai nạn nghiêm trọng", on_click=_load_random_severe_accident, width="stretch")
    if st.session_state.get("load_error"):
        st.warning(st.session_state["load_error"])

    st.subheader("Những yếu tố quan trọng nhất")
    c1, c2, c3 = st.columns(3)
    with c1:
        _select("Light_conditions")
        _select("Type_of_collision")
    with c2:
        st.number_input(vi.col("Number_of_vehicles_involved"), min_value=1, max_value=7, step=1,
                        key=_key("Number_of_vehicles_involved"))
        _select("Cause_of_accident")
    with c3:
        _select("Age_band_of_driver")
        _select("Types_of_Junction")
    st.slider("Giờ xảy ra (0 = nửa đêm, 12 = giữa trưa)", 0, 23, key="hour_pick")

    with st.expander("Thêm chi tiết (người lái, xe, con đường…)"):
        for title, cols in GROUPS.items():
            st.markdown(f"**{title}**")
            columns = st.columns(3)
            for i, col in enumerate(cols):
                with columns[i % 3]:
                    _select(col)

    row = _current_row()
    X = pd.DataFrame([row])[config.RAW_FEATURE_COLS]
    p = float(severity_model.severe_probability(X)[0])
    threshold = severity_model.threshold
    severe = p >= threshold

    st.divider()
    left, right = st.columns([1, 1.3])
    with left:
        st.metric("Khả năng vụ tai nạn nghiêm trọng", shared.pct(p))
        if severe:
            st.error("⚠️ **Cảnh báo: tình huống này có nguy cơ NGHIÊM TRỌNG.**")
        else:
            st.success("✅ **Tình huống này có nguy cơ thấp hơn mức cảnh báo.**")
        if st.session_state.get("loaded") and _unchanged_since_load(row):
            actual = st.session_state["loaded_severity"]
            actual_severe = actual in config.SEVERE_CLASSES
            verdict = "✅ Mô hình đoán đúng." if actual_severe == severe else "❌ Lần này mô hình đoán sai."
            st.info(f"Đây là một vụ tai nạn có thật. Thực tế: **{vi.cls(actual)}**. {verdict}")
        elif st.session_state.get("loaded"):
            st.caption("Bạn đã thay đổi vụ tai nạn có thật, nên không còn so sánh với kết quả thực tế.")
        st.caption(f"Mức cảnh báo của công cụ: {shared.pct(threshold)}. Đây là tín hiệu để xem xét kỹ hơn, "
                   "không phải kết luận chắc chắn.")
    with right:
        st.markdown("**Vì sao mô hình nghĩ như vậy?**")
        contrib = severity_model.explain(X).iloc[0]
        fig = _explanation_chart(contrib, row)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption("Thanh đỏ: yếu tố làm vụ này **nguy hiểm hơn** một vụ bình thường. Thanh xanh: yếu tố "
                   "làm nó **ít nguy hiểm hơn**. Thanh càng dài, ảnh hưởng càng lớn.")

    shared.glossary("Khả năng (xác suất)", (
        "Nếu mô hình nói **30%**, nghĩa là trong rất nhiều vụ tai nạn có hoàn cảnh giống hệt thế này, "
        "khoảng **30 trên 100 vụ** sẽ có người bị thương nặng hoặc tử vong. Nó **không** nói chắc chắn "
        "vụ này sẽ ra sao."
    ))
    st.caption("Đây là công cụ học tập. Kết quả có thể đúng hoặc sai và không được dùng để tự động ra quyết định thật.")
    shared.source_note()
