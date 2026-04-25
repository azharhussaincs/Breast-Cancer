# This is the main entry point for the project, though usually you'll run the other scripts directly.
# It acts as a helpful guide for anyone opening the project for the first time.

import os # To interact with the system if needed

# This block ensures the code only runs if you execute THIS file specifically
if __name__ == "__main__":
    # Welcome message to greet the user
    print("Welcome to the Breast Cancer Diagnostic Project!")
    print("-----------------------------------------------")
    print("Use the following scripts to interact with the project phases:")
    
    # 1. The ML Pipeline: This is where the magic happens (training, math, and saving models)
    print("1. ml_pipeline.py: The Engine Room (Preprocessing, Training, Evaluation)")
    
    # 2. The App: This is the user-friendly interface for making real predictions
    print("2. app.py: The User Interface (Streamlit Dashboard)")
    
    # Simple instructions on how to actually start these scripts from the terminal
    print("\nTo train the models and generate charts:")
    print("   python3 ml_pipeline.py")
    
    print("\nTo launch the interactive dashboard:")
    print("   streamlit run app.py")
    
    # Reminding the user where all the important files (models, plots) are kept
    print("\nNote: All project artifacts (models, plots) are safely stored in the 'artifacts/' folder.")
