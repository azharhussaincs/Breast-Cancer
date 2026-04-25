from pptx import Presentation
from pptx.util import Inches, Pt
import os

def create_presentation():
    prs = Presentation()

    # --- Slide 1: Title ---
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "Breast Cancer Diagnostic Assistant"
    subtitle.text = "Clinical Tool for Breast Mass Classification using Machine Learning\nPresented by: [Your Name]"

    # --- Slide 2: Project Overview ---
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = "Project Overview"
    content = slide.placeholders[1]
    content.text = (
        "- Objective: Classify tumors as Benign or Malignant.\n"
        "- Dataset: Wisconsin Breast Cancer (Diagnostic).\n"
        "- Features: 30 clinical features (Mean, Standard Error, Worst measurements).\n"
        "- Technology Stack: Python, Scikit-Learn, Streamlit, Pandas, Joblib."
    )

    # --- Slide 3: Exploratory Data Analysis (EDA) - Histograms ---
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    title = slide.shapes.title
    title.text = "EDA: Feature Distributions"
    img_path = 'artifacts/histograms.png'
    if os.path.exists(img_path):
        prs.slides[-1].shapes.add_picture(img_path, Inches(1), Inches(1.5), height=Inches(5))

    # --- Slide 4: EDA: Correlation Analysis ---
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    title = slide.shapes.title
    title.text = "EDA: Feature Correlation"
    img_path = 'artifacts/correlation_heatmap.png'
    if os.path.exists(img_path):
        prs.slides[-1].shapes.add_picture(img_path, Inches(1), Inches(1.5), height=Inches(5))

    # --- Slide 5: Data Preprocessing ---
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    title.text = "Data Preprocessing Pipeline"
    content = slide.placeholders[1]
    content.text = (
        "1. Missing Value Check: None found in dataset.\n"
        "2. Duplicate Removal: Verified no duplicate entries.\n"
        "3. Feature Scaling: StandardScaler applied to normalize data.\n"
        "4. Train-Test Split: 80% Training, 20% Testing.\n"
        "5. PCA: Dimensionality reduction for visualization."
    )

    # --- Slide 6: Model Selection & Training ---
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    title.text = "Model Selection & Training"
    content = slide.placeholders[1]
    content.text = (
        "- Models Evaluated:\n"
        "  1. Logistic Regression (GridSearchCV optimized)\n"
        "  2. Random Forest Classifier\n"
        "  3. Ensemble Voting Classifier (Soft Voting)\n"
        "- Optimization: Used cross-validation to ensure generalization."
    )

    # --- Slide 7: Model Performance Metrics ---
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    title.text = "Model Performance (Logistic Regression)"
    content = slide.placeholders[1]
    content.text = (
        "- Accuracy: 97.4%\n"
        "- F1-Score: 0.98\n"
        "- Precision: 0.97\n"
        "- Recall: 0.99\n"
        "- The model shows high reliability for clinical assistance."
    )

    # --- Slide 8: Confusion Matrix ---
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    title = slide.shapes.title
    title.text = "Evaluation: Confusion Matrix"
    img_path = 'artifacts/cm_Logistic_Regression.png'
    if os.path.exists(img_path):
        prs.slides[-1].shapes.add_picture(img_path, Inches(1.5), Inches(1.5), height=Inches(5))

    # --- Slide 9: Streamlit Application ---
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    title.text = "Streamlit Web Application"
    content = slide.placeholders[1]
    content.text = (
        "- Features:\n"
        "  - Interactive sliders for patient parameters.\n"
        "  - Real-time prediction with confidence scores.\n"
        "  - Dynamic visualization of model metrics.\n"
        "  - Professional UI using custom CSS."
    )

    # --- Slide 10: Conclusion ---
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    title.text = "Conclusion"
    content = slide.placeholders[1]
    content.text = (
        "- Successfully developed a high-precision diagnostic tool.\n"
        "- Deployment-ready on Streamlit Community Cloud.\n"
        "- Future Work: Integrate with electronic health records (EHR) and expand dataset."
    )

    prs.save('presentation.pptx')
    print("Presentation saved successfully as presentation.pptx")

if __name__ == "__main__":
    create_presentation()
