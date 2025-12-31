import streamlit as st
import numpy as np
import pickle
from tensorflow import keras
import sys

# Set up the page configuration
st.set_page_config(page_title="🌱 Plant Health Monitor", page_icon="🌿", layout="wide")

@st.cache_resource
def load_model():
    """Loads the model, Scaler, and Label Encoder from files."""
    try:
        # 1. Load the Keras model
        # Use compile=False if the optimizer wasn't saved
        model = keras.models.load_model('plant_health_model.h5', compile=False) 
        
        # 2. Load the Scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        # 3. Load the Label Encoder
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
            
        return model, scaler, le
    except FileNotFoundError as e:
        st.error(f"⚠️ Error: Missing file! Please ensure: plant_health_model.h5, scaler.pkl, and label_encoder.pkl are present.")
        st.error(f"Details: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading files: {e}")
        return None, None, None

model, scaler, le = load_model()
loaded = model is not None

st.title("🌱 Smart Plant Health Monitor")
st.markdown("### AI-Powered Plant Diagnosis System")
st.markdown("---")


if loaded:
    # Add Label Encoder check in the sidebar
    st.sidebar.subheader("ℹ️ Label Encoder Check")
    try:
        # Display the mapping between the number and the name
        le_map = dict(zip(le.transform(le.classes_), le.classes_))
        st.sidebar.write("**Loaded Mapping:**")
        st.sidebar.json(le_map) # Display the map clearly
    except Exception:
        st.sidebar.error("Error displaying LabelEncoder mapping.")
        
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Sensor Readings")
        
        c1, c2 = st.columns(2)
        with c1:
            temp = st.number_input("🌡️ Temperature (°C)", 0.0, 50.0, 24.0, 0.1)
            humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 65.0, 1.0)
            moisture = st.number_input("🌊 Soil Moisture (%)", 0.0, 100.0, 55.0, 1.0)
            ph = st.number_input("⚗️ Soil pH", 0.0, 14.0, 6.5, 0.1)
        
        with c2:
            nutrient = st.number_input("🌿 Nutrient Level", 0.0, 100.0, 50.0, 1.0)
            light = st.number_input("💡 Light Intensity (lux)", 0.0, 100000.0, 15000.0, 100.0)
            health_score = st.number_input("📊 Health Score (0-100)", 0.0, 100.0, 85.0, 1.0)
        
        st.markdown("---")
        predict = st.button("🔍 Diagnose Plant Health", use_container_width=True)
    
    with col2:
        st.subheader("📊 Diagnosis Result")
        
        if predict:
            try:
                # Compile inputs (7 features)
                input_features = [temp, humidity, moisture, ph, nutrient, light, health_score]
                
                # Convert to NumPy array
                X = np.array([input_features])
                
                # Apply the scaler
                X_scaled = scaler.transform(X)
                
                # Perform the prediction
                pred = model.predict(X_scaled, verbose=0)
                
                # Determine the predicted class (highest probability)
                pred_class = np.argmax(pred)
                
                # Get the confidence score
                confidence = pred[0][pred_class] * 100
                
                # Inverse transform to get the text label
                label = str(le.inverse_transform([pred_class])[0])
                
                # --- Improved Display Logic ---
                if 'healthy' in label.lower():
                    st.success(f"### ✅ {label.upper()}")
                    st.balloons()
                else:
                    st.error(f"### ⚠️ {label.upper()}")

                st.metric("Confidence", f"{confidence:.1f}%")
                
                # **Fix for float32 error:** Convert the value to a standard float
                st.progress(float(confidence/100)) 
                
                st.markdown("---")
                st.write("**Prediction Details (Class Probabilities):**")
                
                # Display all class probabilities
                for i, l in enumerate(le.classes_):
                    prob = pred[0][i] * 100
                    st.write(f"**{l}:** {prob:.1f}%")
                    
            except Exception as e:
                st.error(f"❌ General Prediction Error: {e}")
                
        else:
            st.info("👈 Enter sensor values\nand click Diagnose")
            
    # Add example expander (optional, for visual aid)
    with st.expander("📋 Example Values"):
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.markdown("**Healthy Plant:**")
            st.code("Temp: 24°C\nHumidity: 65%\nMoisture: 55%\npH: 6.5\nNutrient: 50\nLight: 15000\nScore: 85")
        with col_ex2:
            st.markdown("**Unhealthy Plant (Dry/Hot):**")
            st.code("Temp: 32°C\nHumidity: 30%\nMoisture: 20%\npH: 5.5\nNutrient: 25\nLight: 3000\nScore: 35")