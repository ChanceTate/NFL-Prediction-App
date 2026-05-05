# NFL Prediction App
Full-stack machine learning application that predicts NFL QB game performance using historical data. Built with Python, scikitlearn, and FastAPI, with a Dockerized backend and interactive frontend for real time predictions

#Overview
This project predicts quarterback performance for upcoming games using historical game data and machine learning models.

This is accomplished by combining:
-A data pipeline for processing QB game logs
-Feature engineering to capture recent performance trends
- Regression models for prediction
-A backend API for real-time prediction
-A simple frontend interface for user interaction
-Docker for reproducible deployment

Quarterback performance varies week-to-week and is context dependent.

This project aims to predict upcoming game QB performance using historical statistics and engineered features representing recent trends.

#Tech Stack
-Python
-pandas
-scikit-learn
-FastAPI
-Docker
-HTML/JavaScript frontend

#Data
Data is source from Pro Football Reference

Each record represents a single QB game and includes:
-Passing yards
-Touchdowns
-Interceptions
-Completions/attempts
-Opponent
-Game date

#Feature engineering
Raw game data is transformed into model-ready features that represent a QB's recent performance.

Key features include:
-Rolling averages (last 3 games)
-Completion percentage trends
-Touchdown and interception rates
-Home vs. Away encoding
-Opponent strength

#Model
This project uses regression  models to predict QB performance

Models used:
-Linear Regression

Targets:
-Passing yards
-Touchdowns
-Interceptions

Models are evaluated using Mean Absolute Error (MAE)

#API
The backend exposes a prediction endpoint using FastAPI

#Frontend
A simple interface allows users to:
-Enter a QB name
-Request prediction
-View projected stats

#Docker
This application is containerized for easy setup and portability