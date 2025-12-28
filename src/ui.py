# D:\Coding\Infosys 6.0\new\src\ui.py

import streamlit as st
import requests
import base64
import io
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

# Backend URL
BACKEND_URL = "http://127.0.0.1:8000/predict"
CONFIG_URL = "http://127.0.0.1:8000/config"

# Page Configuration
st.set_page_config(
    page_title="Oil Spill Detection System",
    page_icon="🛢️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown('<p class="main-header">🛢️ AI-Driven Oil Spill Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time Detection using CNN & U-Net Deep Learning Models</p>', unsafe_allow_html=True)

# Timestamp
current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S IST")
st.caption(f"⏰ Analysis Timestamp: {current_time}")

st.divider()

# Sidebar - Configuration Display
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    try:
        config_response = requests.get(CONFIG_URL, timeout=5)
        if config_response.status_code == 200:
            config = config_response.json()
            st.success("✅ Backend Connected")
            
            st.subheader("Current Thresholds")
            st.metric("CNN Threshold", f"{config['cnn_threshold']*100:.0f}%")
            st.metric("Segmentation Threshold", f"{config['segment_threshold']*100:.0f}%")
            st.metric("Min Oil Area", f"{config['min_oil_area_ratio']*100:.0f}%")
            st.metric("Max Oil Area", f"{config['max_oil_area_ratio']*100:.0f}%")
        else:
            st.error("⚠️ Backend connection failed")
    except:
        st.error("❌ Backend not running")
        st.info("Please start the backend server:\n```bash\nuvicorn be:app --reload --port 8000\n```")
    
    st.divider()
    
    st.subheader("📊 About")
    st.info("""
    **Two-Stage Detection:**
    1. **CNN Classifier** - Detects if oil spill is present
    2. **U-Net Segmentation** - Locates oil spill regions
    
    **Enhanced Features:**
    - Morphological noise filtering
    - False positive validation
    - Connected component analysis
    """)

# Main Content
st.header("📤 Upload Satellite/Marine Image")
st.caption("Supported formats: JPG, JPEG, PNG, TIF")

uploaded = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png", "tif"],
    help="Upload a satellite or marine image to analyze for oil spills"
)

if uploaded:
    # Display uploaded image
    image = Image.open(uploaded).convert("RGB")
    
    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)
    
    st.divider()
    
    # Analyze Button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        analyze_button = st.button("🔍 Analyze Image", type="primary")
    
    if analyze_button:
        with st.spinner("🔄 Analyzing image... Please wait..."):
            try:
                # Send request to backend
                response = requests.post(
                    BACKEND_URL,
                    files={"file": uploaded.getvalue()},
                    timeout=30
                )
                
                if response.status_code != 200:
                    st.error(f"❌ Backend Error: {response.status_code}")
                    st.stop()
                
                result = response.json()
                
                st.divider()
                
                # Display Results
                if result["oil_detected"]:
                    st.success("🚨 OIL SPILL DETECTED!")
                    
                    # Show detection reason if available
                    if "detection_reason" in result:
                        st.info(f"✅ Validation: {result['detection_reason']}")
                else:
                    st.success("✅ NO OIL SPILL DETECTED")
                    
                    # Show rejection reason if available
                    if "rejection_reason" in result:
                        st.warning(f"📋 Reason: {result['rejection_reason']}")
                
                # Metrics Display
                st.subheader("📊 Detection Metrics")
                
                c1, c2, c3, c4 = st.columns(4)
                
                # Status
                with c1:
                    status = "🔴 DETECTED" if result["oil_detected"] else "🟢 NOT DETECTED"
                    st.metric("Oil Spill Status", status)
                
                # Confidence
                with c2:
                    confidence_color = "🟢" if result["confidence"] > 80 else "🟡" if result["confidence"] > 60 else "🔴"
                    st.metric("CNN Confidence", f"{confidence_color} {result['confidence']}%")
                
                # Oil Percentage
                with c3:
                    oil_color = "🔴" if result["oil_percentage"] > 50 else "🟡" if result["oil_percentage"] > 20 else "🟢"
                    st.metric("Oil Spill Coverage", f"{oil_color} {result['oil_percentage']}%")
                
                # Clean Water Percentage
                with c4:
                    water_color = "🟢" if result["non_spill_percentage"] > 80 else "🟡" if result["non_spill_percentage"] > 50 else "🔴"
                    st.metric("Clean Water", f"{water_color} {result['non_spill_percentage']}%")
                
                # Visual Results (only if oil detected)
                if result["oil_detected"] and result["mask_base64"] and result["overlay_base64"]:
                    st.divider()
                    st.subheader("🎨 Visual Analysis")
                    
                    # Decode images
                    mask = Image.open(
                        io.BytesIO(base64.b64decode(result["mask_base64"]))
                    )
                    overlay = Image.open(
                        io.BytesIO(base64.b64decode(result["overlay_base64"]))
                    )
                    
                    # Display in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Original Image**")
                        st.image(image, use_column_width=True)
                    
                    with col2:
                        st.markdown("**Segmentation Mask**")
                        st.image(mask, use_column_width=True)
                        st.caption("White areas indicate detected oil spills")
                    
                    with col3:
                        st.markdown("**Overlay Visualization**")
                        st.image(overlay, use_column_width=True)
                        st.caption("Red overlay shows oil-contaminated regions")
                    
                    # Distribution Chart
                    st.divider()
                    st.subheader("📈 Oil Spill Distribution")
                    
                    col_chart1, col_chart2, col_chart3 = st.columns([1, 2, 1])
                    
                    with col_chart2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        sizes = [result["oil_percentage"], result["non_spill_percentage"]]
                        labels = ["Oil Spill", "Clean Water"]
                        colors = ["#ff4444", "#44ff44"]
                        explode = (0.1, 0)
                        
                        ax.pie(
                            sizes,
                            labels=labels,
                            autopct="%1.2f%%",
                            colors=colors,
                            explode=explode,
                            shadow=True,
                            startangle=90
                        )
                        ax.axis("equal")
                        ax.set_title("Area Distribution Analysis", fontsize=14, fontweight='bold')
                        
                        st.pyplot(fig)
                        plt.close()
                    
                    # Download Section
                    st.divider()
                    st.subheader("💾 Download Results")
                    
                    col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
                    
                    with col_dl1:
                        # Download mask
                        mask_bytes = io.BytesIO()
                        mask.save(mask_bytes, format="PNG")
                        st.download_button(
                            label="📥 Segmentation Mask",
                            data=mask_bytes.getvalue(),
                            file_name=f"oil_mask_{current_time.replace(':', '-')}.png",
                            mime="image/png"
                        )
                    
                    with col_dl2:
                        # Download overlay
                        overlay_bytes = io.BytesIO()
                        overlay.save(overlay_bytes, format="PNG")
                        st.download_button(
                            label="📥 Overlay Image",
                            data=overlay_bytes.getvalue(),
                            file_name=f"oil_overlay_{current_time.replace(':', '-')}.png",
                            mime="image/png"
                        )
                    
                    with col_dl3:
                        # Download chart
                        chart_bytes = io.BytesIO()
                        fig.savefig(chart_bytes, format="PNG", bbox_inches='tight', dpi=150)
                        st.download_button(
                            label="📥 Distribution Chart",
                            data=chart_bytes.getvalue(),
                            file_name=f"oil_chart_{current_time.replace(':', '-')}.png",
                            mime="image/png"
                        )
                    
                    with col_dl4:
                        # Download report (text)
                        report = f"""
Oil Spill Detection Report
==========================
Timestamp: {current_time}

DETECTION RESULTS:
- Status: {'OIL SPILL DETECTED' if result['oil_detected'] else 'NO OIL SPILL'}
- CNN Confidence: {result['confidence']}%
- Oil Coverage: {result['oil_percentage']}%
- Clean Water: {result['non_spill_percentage']}%

VALIDATION:
{result.get('detection_reason', result.get('rejection_reason', 'N/A'))}

SYSTEM CONFIGURATION:
- CNN Threshold: {config.get('cnn_threshold', 'N/A')}
- Segmentation Threshold: {config.get('segment_threshold', 'N/A')}
- Min Oil Area: {config.get('min_oil_area_ratio', 'N/A')}
- Max Oil Area: {config.get('max_oil_area_ratio', 'N/A')}
"""
                        st.download_button(
                            label="📥 Text Report",
                            data=report,
                            file_name=f"oil_report_{current_time.replace(':', '-')}.txt",
                            mime="text/plain"
                        )
                
                # Additional Information
                st.divider()
                with st.expander("ℹ️ Detailed Information"):
                    st.json(result)
            
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Make sure the server is running on port 8000.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>AI-Driven Oil Spill Detection System</strong></p>
    <p>Powered by Deep Learning | CNN + U-Net Architecture</p>
    <p>© 2024 | Environmental Monitoring Solution</p>
</div>
""", unsafe_allow_html=True)