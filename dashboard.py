import streamlit as st
import json
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="AI Context-Aware Mental Health System", layout="wide")

st.markdown("<h1 style='text-align:center;'>🧠 AI Context-Aware Mental Health Monitoring System</h1>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# LOAD DATA
# -------------------------------
def load(user):
    try:
        with open(f"{user}_results.json") as f:
            return json.load(f)
    except:
        return None

user = st.text_input("Enter Patient Name")

if not user:
    st.stop()

data = load(user)

if not data:
    st.warning("No data found")
    st.stop()

sequence = data["emotion_sequence"]

# -------------------------------
# INFO
# -------------------------------
st.markdown(f"### 👤 Patient: {data['user']}")
st.write(f"🕒 {data['time']}")
st.write(f"⏱ Duration: {data['duration_seconds']} sec")

# -------------------------------
# 🎨 STYLE (COLORED CARDS + BUTTON)
# -------------------------------
st.markdown("""
<style>
.card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 20px;
    font-weight: bold;
}
.blue {
    background: linear-gradient(135deg, #2196F3, #42A5F5);
}
.green {
    background: linear-gradient(135deg, #4CAF50, #66BB6A);
}

/* Black Download Button */
.stDownloadButton button {
    background-color: black;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# STRESS LEVEL
# -------------------------------
score = data["mental_health_score"]

if score < 0.3:
    stress = "Low Stress"
elif score < 0.6:
    stress = "Moderate Stress"
else:
    stress = "High Stress"

# -------------------------------
# KPI CARDS (BLUE - GREEN - BLUE)
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.markdown(f'<div class="card blue">Stress Level<br>{stress}</div>', unsafe_allow_html=True)
col2.markdown(f'<div class="card green">Risk<br>{data["status"]}</div>', unsafe_allow_html=True)
col3.markdown(f'<div class="card blue">Stability<br>{data["stability_score"]}</div>', unsafe_allow_html=True)

# ALERT
if stress == "High Stress":
    st.error("🚨 High stress detected! Immediate attention recommended.")

st.markdown("---")

# -------------------------------
# 📉 SMALL GRAPH
# -------------------------------
st.subheader("📉 Emotional Stability Trend")

mapping = {
    "Stable Positive State": 2,
    "Calm Baseline State": 1,
    "Emotional Distress": 0
}

if len(sequence) > 1:
    values = [mapping.get(e, 1) for e in sequence]

    fig, ax = plt.subplots(figsize=(5, 1.5))  # small size
    ax.plot(values, linewidth=2)

    ax.set_yticks([0,1,2])
    ax.set_yticklabels(["Stress","Calm","Stable"])

    st.pyplot(fig)

st.markdown("---")

# -------------------------------
# 🧠 INTERPRETATION
# -------------------------------
st.subheader("🧠 Mental State Interpretation")

st.success(data["interpretation"])
st.write("📊", data["observation"])
st.write("💡", data["recommendation"])

st.markdown("---")

# -------------------------------
# 📄 PDF REPORT
# -------------------------------
st.subheader("📄 Download Report")

def generate_pdf(data, file):
    doc = SimpleDocTemplate(file)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'title',
        parent=styles['Title'],
        alignment=1,
        textColor=colors.darkblue,
        fontSize=18
    )

    heading_style = ParagraphStyle(
        'heading',
        parent=styles['Heading2'],
        spaceAfter=6
    )

    normal = styles['Normal']

    content = []

    content.append(Paragraph("AI Mental Health Analysis Report", title_style))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"<b>Patient:</b> {data['user']}", normal))
    content.append(Paragraph(f"<b>Time:</b> {data['time']}", normal))
    content.append(Paragraph(f"<b>Duration:</b> {data['duration_seconds']} sec", normal))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Mental State Interpretation", heading_style))
    content.append(Paragraph(data["interpretation"], normal))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Observation", heading_style))
    content.append(Paragraph(data["observation"], normal))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Recommendation", heading_style))
    content.append(Paragraph(data["recommendation"], normal))

    doc.build(content)

pdf_file = f"{user}_report.pdf"
generate_pdf(data, pdf_file)

with open(pdf_file, "rb") as f:
    st.download_button("Download PDF", f, file_name=pdf_file)