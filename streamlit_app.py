if submitted:
    try:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        confidence = max(probability)

        # Determine readable label
        unique_values = df[target_col].unique()
        if len(unique_values) == 2 and set(unique_values) == {0, 1}:
            if prediction == 1:
                label = "⚠️ High Risk: Cancer may recur"
                short_label = "Recurrence Likely"
                result_class = "prediction-positive"
            else:
                label = "✅ Low Risk: Cancer not likely to recur"
                short_label = "No Recurrence Expected"
                result_class = "prediction-negative"
        else:
            label = f"🔍 Predicted Class: {prediction}"
            short_label = str(prediction)
            result_class = "prediction-result"

        st.markdown(f"""
        <div class="prediction-result {result_class}">
            <h1>🔮 Prediction Result</h1>
            <h2 style="margin: 1rem 0;">{label}</h2>
            <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; margin-top: 2rem;">
                <div>
                    <h3>Confidence Level</h3>
                    <h1>{confidence:.1%}</h1>
                </div>
                <div>
                    <h3>Risk Category</h3>
                    <h1>{'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'}</h1>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show probabilities
        if len(probability) == 2:
            st.markdown("### 📈 Prediction Probabilities")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🟢 No Recurrence", f"{probability[0]:.1%}")
            with col2:
                st.metric("🔴 Recurrence", f"{probability[1]:.1%}")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
