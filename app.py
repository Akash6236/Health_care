import os
import random
import sqlite3
from datetime import date, timedelta
from functools import wraps
from pathlib import Path

import pandas as pd
from flask import Flask, abort, flash, g, redirect, render_template, request, session, url_for
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "hospital_readmission.db"
DEPARTMENTS = ["Cardiology", "General Medicine", "Neurology", "Orthopedics", "Pulmonology"]
FOLLOW_UP_STATUSES = ["Completed", "Pending", "Missed", "Not scheduled"]
RISK_DISCLAIMER = (
    "Readmission risk predictions are generated for academic demonstration using fictional data. "
    "They are not medical diagnoses and must not be used for clinical decision-making."
)
PROJECT_BOUNDARY = (
    "This project is a prototype healthcare analytics system developed for academic purposes. "
    "It uses fictional patient data to demonstrate patient-flow analysis, readmission tracking and "
    "basic machine-learning-based risk estimation. It is not intended for clinical use, real-world "
    "medical decision-making, or deployment within a production hospital environment."
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "academic-demo-session-key")
DEMO_PASSWORD_HASH = generate_password_hash("admin123")


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def seed_patients():
    rng = random.Random(2026)
    first_names = [
        "Arjun", "Rohan", "Karan", "Siddharth", "Aryan",
        "Kabir", "Veer", "Ishaan", "Aditya", "Rishi",
        "Rahul", "Nikhil", "Vikram", "Amit", "Raj",
        "Sanjay", "Manoj", "Pankaj", "Gaurav", "Varun",
        "Abhishek", "Naveen", "Rakesh", "Deepak", "Ajay",
    ]
    last_names = [
        "Sharma", "Verma", "Gupta", "Kumar", "Singh",
        "Patel", "Rao", "Reddy", "Joshi", "Das",
        "Nair", "Iyer", "Chatterjee", "Banerjee", "Mukherjee",
        "Malhotra", "Kapoor", "Khanna", "Chopra", "Arora",
        "Yadav", "Chauhan", "Rathore", "Jain", "Bajaj",
    ]
    neighborhoods = [
        "Green Park", "Andheri West", "Koramangala", "Jubilee Hills", "Mylapore",
        "Salt Lake", "Bodakdev", "T Nagar", "Park Street", "Banjara Hills",
    ]
    city_state = [
        ("Delhi", "Delhi"),
        ("Mumbai", "Maharashtra"),
        ("Bangalore", "Karnataka"),
        ("Hyderabad", "Telangana"),
        ("Chennai", "Tamil Nadu"),
        ("Kolkata", "West Bengal"),
        ("Pune", "Maharashtra"),
        ("Ahmedabad", "Gujarat"),
        ("Jaipur", "Rajasthan"),
        ("Lucknow", "Uttar Pradesh"),
        ("Kanpur", "Uttar Pradesh"),
        ("Nagpur", "Maharashtra"),
        ("Indore", "Madhya Pradesh"),
        ("Bhopal", "Madhya Pradesh"),
        ("Patna", "Bihar"),
    ]
    diagnoses = [
        ("Hypertension", "Cardiology"),
        ("Type 2 Diabetes", "General Medicine"),
        ("Acute Coronary Syndrome", "Cardiology"),
        ("Community-Acquired Pneumonia", "Pulmonology"),
        ("Dengue Fever", "General Medicine"),
        ("Malaria", "General Medicine"),
        ("Typhoid Fever", "General Medicine"),
        ("Tuberculosis", "Pulmonology"),
        ("Acute Gastroenteritis", "General Medicine"),
        ("Cholelithiasis", "General Medicine"),
        ("Asthma", "Pulmonology"),
        ("COPD", "Pulmonology"),
        ("Cerebrovascular Accident", "Neurology"),
        ("Acute Pancreatitis", "General Medicine"),
        ("Septicemia", "General Medicine"),
        ("Dengue Hemorrhagic Fever", "General Medicine"),
        ("Japanese Encephalitis", "Neurology"),
        ("Leptospirosis", "General Medicine"),
        ("Gout", "General Medicine"),
        ("Rheumatoid Arthritis", "General Medicine"),
        ("Fracture of Right Femur", "Orthopedics"),
        ("Burns", "General Medicine"),
        ("Appendicitis", "General Medicine"),
        ("Ectopic Pregnancy", "General Medicine"),
    ]
    start = date(2024, 1, 10)
    patients = []
    for index in range(25):
        name = f"{first_names[index % len(first_names)]} {last_names[(index * 3 + index // 4) % len(last_names)]}"
        phone = f"{rng.randint(6000000000, 9999999999)}"
        city, state = city_state[index % len(city_state)]
        address = f"{rng.randint(1, 99)}, {neighborhoods[index % len(neighborhoods)]}, {city}, {state}"
        age = [22, 28, 35, 43, 52, 61, 69, 75, 84][index % 9]
        diagnosis, department = diagnoses[index % len(diagnoses)]
        previous_admissions = (index * 2 + index // 5) % 4
        length_of_stay = [2, 3, 4, 5, 6, 7, 9, 11, 14][(index * 2 + index // 3) % 9]
        follow_up = FOLLOW_UP_STATUSES[(index * 3 + index // 6) % len(FOLLOW_UP_STATUSES)]
        risk_signal = (
            int(age >= 65)
            + int(previous_admissions >= 2)
            + int(length_of_stay >= 9)
            + int(follow_up in ("Missed", "Not scheduled"))
        )
        readmitted = risk_signal >= 2 or (risk_signal == 1 and index % 3 == 0)
        admission = start + timedelta(days=index * 12 + rng.randint(0, 5))
        discharge = admission + timedelta(days=length_of_stay)
        readmission = discharge + timedelta(days=rng.randint(18, 75)) if readmitted else None
        patients.append(
            (
                name,
                phone,
                address,
                age,
                department,
                diagnosis,
                admission.isoformat(),
                discharge.isoformat(),
                readmission.isoformat() if readmission else None,
                previous_admissions,
                follow_up,
            )
        )
    return patients


def init_db():
    db = sqlite3.connect(DATABASE)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT NOT NULL,
            address TEXT NOT NULL,
            age INTEGER NOT NULL CHECK (age BETWEEN 0 AND 120),
            department TEXT NOT NULL,
            diagnosis TEXT NOT NULL,
            admission_date TEXT NOT NULL,
            discharge_date TEXT,
            readmission_date TEXT,
            previous_admissions INTEGER NOT NULL DEFAULT 0 CHECK (previous_admissions >= 0),
            follow_up_status TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS app_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    seeded = db.execute("SELECT value FROM app_meta WHERE key = 'synthetic_seeded'").fetchone()
    if seeded is None:
        db.executemany(
            """INSERT INTO patients
               (name, phone, address, age, department, diagnosis, admission_date,
                discharge_date, readmission_date, previous_admissions, follow_up_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            seed_patients(),
        )
        db.execute("INSERT INTO app_meta (key, value) VALUES ('synthetic_seeded', 'yes')")
    db.commit()
    db.close()


def patient_dict(row):
    patient = dict(row)
    admission = date.fromisoformat(patient["admission_date"])
    discharge = patient["discharge_date"]
    patient["length_of_stay"] = (
        (date.fromisoformat(discharge) - admission).days if discharge else None
    )
    return patient


def all_patients():
    rows = get_db().execute("SELECT * FROM patients ORDER BY admission_date DESC, id DESC").fetchall()
    return [patient_dict(row) for row in rows]


def get_patient(patient_id):
    row = get_db().execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    if row is None:
        abort(404)
    return patient_dict(row)


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped_view


def summary_stats(patients):
    discharged = [p for p in patients if p["discharge_date"]]
    returned = [p for p in discharged if p["readmission_date"]]
    lengths = [p["length_of_stay"] for p in discharged]
    return {
        "admitted": len(patients),
        "discharged": len(discharged),
        "readmitted": len(returned),
        "readmission_rate": round(len(returned) / len(discharged) * 100, 1) if discharged else 0,
        "average_stay": round(sum(lengths) / len(lengths), 1) if lengths else 0,
    }


def department_stats(patients):
    result = []
    for department in DEPARTMENTS:
        discharged = [
            p for p in patients
            if p["department"] == department and p["discharge_date"]
        ]
        returned = [p for p in discharged if p["readmission_date"]]
        result.append(
            {
                "department": department,
                "discharged": len(discharged),
                "readmitted": len(returned),
                "rate": round(len(returned) / len(discharged) * 100, 1) if discharged else 0,
                "average_stay": round(
                    sum(p["length_of_stay"] for p in discharged) / len(discharged), 1
                ) if discharged else 0,
            }
        )
    return result


def month_labels(count=6):
    today = date.today().replace(day=1)
    months = []
    for offset in range(count - 1, -1, -1):
        month = today.month - offset
        year = today.year
        while month <= 0:
            month += 12
            year -= 1
        months.append(date(year, month, 1))
    return months


def monthly_trends(patients):
    events = [
        date.fromisoformat(patient[field])
        for patient in patients
        for field in ("admission_date", "readmission_date")
        if patient[field]
    ]
    months = month_labels() if not events else month_labels_ending(events)
    labels = [month.strftime("%b %Y") for month in months]
    admissions = [0] * len(months)
    readmissions = [0] * len(months)
    for patient in patients:
        admitted = date.fromisoformat(patient["admission_date"])
        for index, month in enumerate(months):
            if admitted.year == month.year and admitted.month == month.month:
                admissions[index] += 1
                break
        if patient["readmission_date"]:
            returned = date.fromisoformat(patient["readmission_date"])
            for index, month in enumerate(months):
                if returned.year == month.year and returned.month == month.month:
                    readmissions[index] += 1
                    break
    return {
        "labels": labels,
        "datasets": [
            {"label": "Admissions", "data": admissions, "color": "#176b87"},
            {"label": "Recorded readmissions", "data": readmissions, "color": "#de8246"},
        ],
    }


def month_labels_ending(events, count=6):
    latest = max(events).replace(day=1)
    months = []
    for offset in range(count - 1, -1, -1):
        month = latest.month - offset
        year = latest.year
        while month <= 0:
            month += 12
            year -= 1
        months.append(date(year, month, 1))
    return months


def follow_up_stats(patients):
    result = []
    for status in FOLLOW_UP_STATUSES:
        group = [
            p for p in patients
            if p["follow_up_status"] == status and p["discharge_date"]
        ]
        returned = sum(bool(p["readmission_date"]) for p in group)
        result.append({"status": status, "count": len(group), "readmitted": returned})
    return result


def build_risk_results(patients):
    eligible = [p for p in patients if p["discharge_date"]]
    frame = pd.DataFrame(eligible)
    outcomes = frame["readmission_date"].notna().astype(int) if not frame.empty else pd.Series(dtype=int)
    if len(frame) < 8 or outcomes.nunique() < 2 or outcomes.value_counts().min() < 2:
        return [], None

    numeric_features = ["age", "length_of_stay", "previous_admissions"]
    categorical_features = ["department", "follow_up_status"]
    feature_columns = numeric_features + categorical_features
    frame["observed_readmission"] = outcomes
    x = frame[feature_columns]
    y = frame["observed_readmission"]

    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    model = Pipeline(
        [
            ("preprocessing", transformer),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    evaluation_accuracy = None
    if train_y.nunique() == 2 and test_y.nunique() == 2:
        model.fit(train_x, train_y)
        evaluation_accuracy = round(accuracy_score(test_y, model.predict(test_x)) * 100, 1)
    model.fit(x, y)
    probabilities = model.predict_proba(x)[:, list(model.classes_).index(1)]
    output = []
    for patient, probability in zip(eligible, probabilities):
        score = round(float(probability) * 100, 1)
        category = "High" if score >= 65 else "Medium" if score >= 35 else "Low"
        output.append({**patient, "risk_score": score, "risk_category": category})
    output.sort(key=lambda item: item["risk_score"], reverse=True)
    return output, evaluation_accuracy


def readmission_definition():
    return (
        "A readmission is a recorded return after discharge with a valid readmission date. "
        "No recorded return means none appears in this dataset; it does not mean the patient "
        "will never return."
    )


def collect_insights(patients):
    stats = summary_stats(patients)
    departments = department_stats(patients)
    followups = follow_up_stats(patients)
    insights = [
        f"{stats['readmitted']} of {stats['discharged']} discharged records have a subsequent "
        f"return recorded (readmission rate: {stats['readmission_rate']}%)."
    ]
    if stats["discharged"]:
        insights.append(f"The average recorded treatment period is {stats['average_stay']} days.")
    populated = [item for item in departments if item["discharged"]]
    if populated:
        highest = max(populated, key=lambda item: item["rate"])
        insights.append(
            f"{highest['department']} has the highest observed department rate "
            f"({highest['rate']}%, {highest['readmitted']} of {highest['discharged']} discharged records)."
        )
    groups = [item for item in followups if item["count"]]
    if groups:
        completed = next((item for item in groups if item["status"] == "Completed"), None)
        not_completed = [
            item for item in groups if item["status"] in ("Missed", "Pending", "Not scheduled")
        ]
        other_total = sum(item["count"] for item in not_completed)
        other_returns = sum(item["readmitted"] for item in not_completed)
        if completed and other_total:
            completed_rate = completed["readmitted"] / completed["count"] * 100
            other_rate = other_returns / other_total * 100
            insights.append(
                f"Recorded returns are {completed_rate:.1f}% among completed follow-ups and "
                f"{other_rate:.1f}% among other follow-up statuses in this synthetic sample; "
                "this is descriptive and does not establish causation."
            )
    return insights


def form_values():
    return {
        "name": request.form.get("name", "").strip(),
        "phone": request.form.get("phone", "").strip(),
        "address": request.form.get("address", "").strip(),
        "age": request.form.get("age", "").strip(),
        "department": request.form.get("department", "").strip(),
        "diagnosis": request.form.get("diagnosis", "").strip(),
        "admission_date": request.form.get("admission_date", "").strip(),
        "discharge_date": request.form.get("discharge_date", "").strip(),
        "readmission_date": request.form.get("readmission_date", "").strip(),
        "previous_admissions": request.form.get("previous_admissions", "0").strip(),
        "follow_up_status": request.form.get("follow_up_status", "").strip(),
    }


def validate_patient(values):
    errors = []
    required = ["name", "phone", "address", "age", "department", "diagnosis", "admission_date"]
    if any(not values[key] for key in required):
        errors.append("Complete all required fields.")
    try:
        age = int(values["age"])
        if not 0 <= age <= 120:
            errors.append("Age must be between 0 and 120.")
    except ValueError:
        errors.append("Age must be a whole number.")
    try:
        previous = int(values["previous_admissions"])
        if previous < 0:
            errors.append("Previous admissions cannot be negative.")
    except ValueError:
        errors.append("Previous admissions must be a whole number.")
    if values["department"] not in DEPARTMENTS:
        errors.append("Choose a listed department.")
    if values["follow_up_status"] not in FOLLOW_UP_STATUSES:
        errors.append("Choose a listed follow-up status.")

    parsed = {}
    for key in ("admission_date", "discharge_date", "readmission_date"):
        if values[key]:
            try:
                parsed[key] = date.fromisoformat(values[key])
            except ValueError:
                errors.append(f"Enter a valid {key.replace('_', ' ')}.")
    if "admission_date" in parsed and "discharge_date" in parsed:
        if parsed["discharge_date"] < parsed["admission_date"]:
            errors.append("Discharge date cannot be before admission date.")
    if "readmission_date" in parsed:
        if "discharge_date" not in parsed:
            errors.append("A readmission date requires a discharge date.")
        elif parsed["readmission_date"] <= parsed["discharge_date"]:
            errors.append("Readmission date must be after discharge.")
    return errors


def save_patient(values, patient_id=None):
    fields = (
        values["name"], values["phone"], values["address"], int(values["age"]),
        values["department"], values["diagnosis"], values["admission_date"],
        values["discharge_date"] or None, values["readmission_date"] or None,
        int(values["previous_admissions"]), values["follow_up_status"],
    )
    db = get_db()
    if patient_id is None:
        db.execute(
            """INSERT INTO patients
               (name, phone, address, age, department, diagnosis, admission_date,
                discharge_date, readmission_date, previous_admissions, follow_up_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            fields,
        )
    else:
        db.execute(
            """UPDATE patients SET name=?, phone=?, address=?, age=?, department=?, diagnosis=?,
               admission_date=?, discharge_date=?, readmission_date=?, previous_admissions=?,
               follow_up_status=? WHERE id=?""",
            (*fields, patient_id),
        )
    db.commit()


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == "admin" and check_password_hash(DEMO_PASSWORD_HASH, password):
            session.clear()
            session["logged_in"] = True
            target = request.args.get("next", "")
            if not target.startswith("/") or target.startswith("//"):
                target = url_for("dashboard")
            return redirect(target)
        flash("Username or password was not recognized.", "error")
    return render_template("login.html")


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/")
@login_required
def dashboard():
    patients = all_patients()
    stats = summary_stats(patients)
    departments = department_stats(patients)
    return render_template(
        "dashboard.html",
        stats=stats,
        departments=departments,
        trend_chart=monthly_trends(patients),
        department_chart={
            "labels": [item["department"] for item in departments],
            "datasets": [
                {
                    "label": "Observed readmission rate (%)",
                    "data": [item["rate"] for item in departments],
                    "color": "#176b87",
                }
            ],
        },
        definition=readmission_definition(),
    )


@app.get("/patients")
@login_required
def patients_page():
    patients = all_patients()
    search = request.args.get("q", "").strip().casefold()
    department = request.args.get("department", "")
    outcome = request.args.get("outcome", "")
    if search:
        patients = [
            p for p in patients
            if search in " ".join(
                str(p[key]) for key in ("name", "phone", "diagnosis", "department")
            ).casefold()
        ]
    if department in DEPARTMENTS:
        patients = [p for p in patients if p["department"] == department]
    if outcome == "readmitted":
        patients = [p for p in patients if p["readmission_date"]]
    elif outcome == "no-recorded-return":
        patients = [
            p for p in patients if p["discharge_date"] and not p["readmission_date"]
        ]
    elif outcome == "active":
        patients = [p for p in patients if not p["discharge_date"]]
    return render_template(
        "patients.html",
        patients=patients,
        departments=DEPARTMENTS,
        selected_department=department,
        selected_outcome=outcome,
        search=request.args.get("q", ""),
    )


@app.route("/patients/new", methods=["GET", "POST"])
@login_required
def patient_new():
    values = form_values() if request.method == "POST" else {}
    if request.method == "POST":
        errors = validate_patient(values)
        if errors:
            for error in errors:
                flash(error, "error")
        else:
            save_patient(values)
            flash("Fictional/demo patient record added.", "success")
            return redirect(url_for("patients_page"))
    return render_template(
        "patient_form.html", patient=values, departments=DEPARTMENTS,
        follow_up_statuses=FOLLOW_UP_STATUSES, mode="Add",
    )


@app.route("/patients/<int:patient_id>")
@login_required
def patient_detail(patient_id):
    return render_template("patient_detail.html", patient=get_patient(patient_id))


@app.route("/patients/<int:patient_id>/edit", methods=["GET", "POST"])
@login_required
def patient_edit(patient_id):
    patient = get_patient(patient_id)
    values = form_values() if request.method == "POST" else patient
    if request.method == "POST":
        errors = validate_patient(values)
        if errors:
            for error in errors:
                flash(error, "error")
        else:
            save_patient(values, patient_id)
            flash("Patient record updated.", "success")
            return redirect(url_for("patient_detail", patient_id=patient_id))
    return render_template(
        "patient_form.html", patient=values, departments=DEPARTMENTS,
        follow_up_statuses=FOLLOW_UP_STATUSES, mode="Edit",
    )


@app.post("/patients/<int:patient_id>/delete")
@login_required
def patient_delete(patient_id):
    get_patient(patient_id)
    get_db().execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    get_db().commit()
    flash("Patient record deleted.", "success")
    return redirect(url_for("patients_page"))


@app.get("/analytics")
@login_required
def analytics():
    patients = all_patients()
    departments = department_stats(patients)
    followups = follow_up_stats(patients)
    return render_template(
        "analytics.html",
        stats=summary_stats(patients),
        departments=departments,
        followups=followups,
        department_chart={
            "labels": [item["department"] for item in departments],
            "datasets": [
                {"label": "Discharged", "data": [item["discharged"] for item in departments], "color": "#176b87"},
                {"label": "Readmitted", "data": [item["readmitted"] for item in departments], "color": "#de8246"},
            ],
        },
        followup_chart={
            "labels": [item["status"] for item in followups],
            "datasets": [
                {"label": "Discharged records", "data": [item["count"] for item in followups], "color": "#477a56"},
                {"label": "With recorded return", "data": [item["readmitted"] for item in followups], "color": "#c45b4d"},
            ],
        },
        definition=readmission_definition(),
    )


@app.get("/risk")
@login_required
def risk():
    predictions, accuracy = build_risk_results(all_patients())
    factors = [
        "Age",
        "Length of stay",
        "Number of previous admissions",
        "Department",
        "Follow-up status",
    ]
    return render_template(
        "risk.html",
        predictions=predictions,
        accuracy=accuracy,
        factors=factors,
        disclaimer=RISK_DISCLAIMER,
    )


@app.get("/insights")
@login_required
def insights():
    patients = all_patients()
    return render_template(
        "insights.html",
        insights=collect_insights(patients),
        stats=summary_stats(patients),
        definition=readmission_definition(),
    )


@app.context_processor
def template_context():
    return {"project_boundary": PROJECT_BOUNDARY}


init_db()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
