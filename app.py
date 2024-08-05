import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model and encoders
model = joblib.load('saved_model/model.pk1')
label_encoders = joblib.load('saved_model/label_encoders.pkl')

select_features = [
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'count',
    'same_srv_rate',
    'diff_srv_rate',
    'dst_host_srv_count',
    'dst_host_same_srv_rate'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        if 'features' not in data:
            return jsonify({'error': "'features' key is missing from the request payload."}), 400

        # Convert the features list to DataFrame
        df = pd.DataFrame(data['features'])

        # Ensure that the DataFrame contains all required features
        missing_features = [feature for feature in select_features if feature not in df.columns]
        if missing_features:
            return jsonify({'error': f"Missing features: {', '.join(missing_features)}."}), 400
        
        # Select only the required features
        df_selected = df[select_features]

        # Encode categorical features
        for col in select_features:
            if col in df_selected.select_dtypes(include=['object']).columns:
                if col in label_encoders:
                    encoder = label_encoders.get(col)
                    if encoder:
                        if hasattr(encoder, 'classes_'):
                            df_selected[col] = encoder.transform(df_selected[col])
                        else:
                            return jsonify({'error': f"Encoder for column '{col}' is not fitted."}), 400
                    else:
                        return jsonify({'error': f"Encoder for column '{col}' not found."}), 400
                else:
                    return jsonify({'error': f"LabelEncoder for column '{col}' does not exist."}), 400

        features_array = df_selected.values
        prediction = model.predict(features_array)
        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)
