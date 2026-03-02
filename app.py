from flask import Flask, render_template, request, jsonify
import pickle, os, numpy as np

app = Flask(__name__)


class HypertensionModel:
    def __init__(self, model, scaler, features):
        self.model   = model
        self.scaler  = scaler
        self.features= features
        self.stage_labels = {0:'Normal',1:'Stage 1 Hypertension',
                             2:'Stage 2 Hypertension',3:'Hypertensive Crisis'}
        self.encode_maps = {
            'Gender':{'Male':0,'Female':1},
            'Age':{'18-34':0,'35-50':1,'51-64':2,'65+':3},
            'Severity':{'None':0,'Mild':1,'Moderate':2,'Sever':3,'Severe':3},
            'Whendiagnoused':{'<1 Year':0,'1 - 5 Years':1,'>5 Years':2},
            'Systolic':{'100+':0,'111 - 120':1,'121- 130':2,'121 - 130':2,'130+':3},
            'Diastolic':{'70 - 80':0,'81 - 90':1,'91 - 100':2,'100+':3,'130+':4},
        }
        self.binary_cols=['History','Patient','TakeMedication','BreathShortness',
                          'VisualChanges','NoseBleeding','ControlledDiet']
    def predict_patient(self, data):
        row=[]
        for feat in self.features:
            val=str(data.get(feat,'')).strip()
            if feat in self.encode_maps: val=self.encode_maps[feat].get(val,0)
            elif feat in self.binary_cols: val=1 if val in ['Yes','yes','1'] else 0
            else: val=0
            row.append(float(val))
        X_sc=self.scaler.transform(np.array(row).reshape(1,-1))
        stage=int(self.model.predict(X_sc)[0])
        proba=self.model.predict_proba(X_sc)[0]
        return {'stage':stage,'label':self.stage_labels[stage],
                'confidence':round(float(max(proba))*100,1),
                'probabilities':{self.stage_labels[i]:round(float(p)*100,1)
                                 for i,p in enumerate(proba)}}
    def predict(self,X): return self.model.predict(self.scaler.transform(X))
    def predict_proba(self,X): return self.model.predict_proba(self.scaler.transform(X))

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'logreg_model.pkl')
model = None

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded. Features:", getattr(model, 'features', 'N/A'))
except FileNotFoundError:
    print("⚠️  logreg_model.pkl not found. Falling back to rule-based mode.")

# ── Stage metadata ───────────────────────────────────────────
STAGES = {
    0: {"label": "Normal",                "color": "#4CAF50", "urgency": "low",
        "summary": "Blood pressure is within healthy range."},
    1: {"label": "Stage 1 Hypertension",  "color": "#FFB347", "urgency": "medium",
        "summary": "Elevated pressure — lifestyle changes recommended."},
    2: {"label": "Stage 2 Hypertension",  "color": "#FF6B35", "urgency": "high",
        "summary": "High BP — medical evaluation and likely medication needed."},
    3: {"label": "Hypertensive Crisis",   "color": "#FF2D2D", "urgency": "critical",
        "summary": "URGENT: Seek emergency medical care immediately."},
}

def rule_based_stage(systolic_range: str, diastolic_range: str) -> int:
    sys_map = {'< 120': 0, '120-129': 1, '130-139': 2,
               '140-159': 3, '160-179': 4, '180+': 5}
    dia_map = {'< 80': 0, '80-89': 1, '90-99': 2, '100-109': 3, '110+': 4}
    s = sys_map.get(systolic_range, 2)
    d = dia_map.get(diastolic_range, 1)
    if s >= 5 or d >= 4: return 3
    if s >= 3 or d >= 2: return 2
    if s == 2 or d == 1: return 1
    if s == 1:           return 1
    return 0


def get_recommendations(stage: int, data: dict) -> list:
    recs = []
    if stage == 3:
        recs.append({"icon": "🚨", "text": "URGENT: Go to the emergency room or call emergency services now.", "priority": "critical"})
    if stage >= 2:
        recs.append({"icon": "💊", "text": "Discuss antihypertensive medication with your physician immediately.", "priority": "high"})
    if stage >= 1:
        recs.append({"icon": "🥗", "text": "Follow the DASH diet: limit sodium to under 2,300 mg/day.", "priority": "high"})
        recs.append({"icon": "🏃", "text": "Target 150 minutes of moderate aerobic exercise per week.", "priority": "medium"})
        recs.append({"icon": "📊", "text": "Monitor blood pressure daily and keep a written log.", "priority": "medium"})
    severity = data.get('symptom_severity', 'None')
    if severity == 'Severe':
        recs.append({"icon": "⚠️",  "text": "Severe symptoms present — do not delay medical attention.", "priority": "high"})
    if data.get('shortness_breath') == 'Yes':
        recs.append({"icon": "😮‍💨", "text": "Shortness of breath may indicate cardiac strain — consult a doctor promptly.", "priority": "high"})
    if data.get('visual_changes') == 'Yes':
        recs.append({"icon": "👁️",  "text": "Visual disturbances can signal hypertensive retinopathy — eye exam recommended.", "priority": "high"})
    if data.get('nosebleeds') == 'Yes':
        recs.append({"icon": "🩸",  "text": "Frequent nosebleeds may be linked to elevated blood pressure.", "priority": "medium"})
    if data.get('controlled_diet') == 'No':
        recs.append({"icon": "🥦", "text": "Adopting a controlled, low-sodium diet can reduce systolic BP by 8-14 mmHg.", "priority": "medium"})
    if data.get('family_history') == 'Yes':
        recs.append({"icon": "🧬", "text": "Family history increases risk — annual cardiovascular screening is advised.", "priority": "medium"})
    if data.get('age_group') == '65+':
        recs.append({"icon": "👴", "text": "Older adults: rise slowly from seated positions to avoid orthostatic hypotension.", "priority": "low"})
    if stage == 0:
        recs.append({"icon": "✅", "text": "Great news! Maintain your healthy lifestyle and get annual BP check-ups.", "priority": "low"})
    return recs


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        systolic_range  = data.get('systolic',  '< 120')
        diastolic_range = data.get('diastolic', '< 80')
        confidence = None
        proba_map = None

        if model is not None:
            try:
                result = model.predict_patient(data)
                stage = result['stage']
                confidence = result['confidence']
                proba_map = result['probabilities']
            except Exception as e:
                print(f"Model error: {e}, using rules.")
                stage = rule_based_stage(systolic_range, diastolic_range)
        else:
            stage = rule_based_stage(systolic_range, diastolic_range)

        stage_info = STAGES[stage]
        recs = get_recommendations(stage, data)
        risk_score = round((stage / 3) * 100)

        return jsonify({
            'success': True,
            'stage': stage,
            'label': stage_info['label'],
            'color': stage_info['color'],
            'urgency': stage_info['urgency'],
            'summary': stage_info['summary'],
            'confidence': confidence,
            'risk_score': risk_score,
            'systolic': systolic_range,
            'diastolic': diastolic_range,
            'probabilities': proba_map,
            'recommendations': recs,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None,
                    'features': getattr(model, 'features', None)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
