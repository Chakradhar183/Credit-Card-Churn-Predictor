# Deploying to Streamlit Cloud

This guide will help you deploy your Credit Card Churn Prediction app to Streamlit Cloud.

## Prerequisites

1.  **GitHub Account**: You need a GitHub account to host your code.
2.  **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io/) using your GitHub account.

## Step 1: Prepare Your Project (Already Done)

I have already:
-   Created a `requirements.txt` file with all necessary dependencies.
-   Updated `.gitignore` to ensure your trained models (`models/*.pkl`) are uploaded to GitHub.

## Step 2: Push to GitHub

You need to push your local code to a new GitHub repository.

1.  **Create a new repository** on GitHub (e.g., `credit-card-churn-app`).
    -   Do **not** initialize it with a README, .gitignore, or license (you already have them).

2.  **Open your terminal** (Git Bash or Command Prompt) in the project folder:
    ```bash
    cd c:/Users/sethu/Downloads/CustomerChurn
    ```

3.  **Initialize Git and push**:
    ```bash
    git init
    git add .
    git commit -m "Initial commit for Streamlit deployment"
    git branch -M main
    git remote add origin https://github.com/<YOUR-USERNAME>/credit-card-churn-app.git
    git push -u origin main
    ```
    *Replace `<YOUR-USERNAME>` with your actual GitHub username.*

## Step 3: Deploy on Streamlit Cloud

1.  **Log in** to [Streamlit Cloud](https://share.streamlit.io/).
2.  Click **"New app"**.
3.  Select **"Use existing repo"**.
4.  **Select your repository**: e.g., `<your-username>/credit-card-churn-app`.
5.  **Branch**: `main`.
6.  **Main file path**: `app/app.py`.
7.  Click **"Deploy!"**.

## Step 4: Watch it Build

Streamlit Cloud will now install the dependencies from `requirements.txt` and launch your app. This may take a few minutes.

## Troubleshooting

-   **"Module not found"**: Check if the missing module is listed in `requirements.txt`.
-   **"Model not found"**: Ensure `models/best_model.pkl` and `models/preprocessor.pkl` were successfully pushed to GitHub. Check your repo online to confirm.
