import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

data = pd.read_csv('accident.csv')
print(data.head())

#independent variable
X = data.drop('AccidentSeverity', axis=1)
# Define dependent variable (target)
y = data['AccidentSeverity']

categorical_features = ['WeatherConditions', 'TimeOfDay', 'DayOfWeek', 'RoadConditions', 'LocationType', 'VehicleType', 'UseOfSeatbelts']
numerical_features = ['DriverAge', 'DriverExperience', 'TrafficVolume']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

score = pipeline.score(X_test, y_test)
print(f'Model fit : {score:.2f}')

hypothetical_input = pd.DataFrame({
    'WeatherConditions': ['Clear'],
    'TimeOfDay': ['Morning'],
    'DayOfWeek': ['Weekday'],
    'RoadConditions': ['Dry'],
    'LocationType': ['Urban'],
    'VehicleType': ['Car'],
    'UseOfSeatbelts': ['Yes'],
    'DriverAge': [30],
    'DriverExperience': [5],
    'TrafficVolume': [60]
})
predicted_severity = pipeline.predict(hypothetical_input)
print(f'Accident severity: {predicted_severity[0]:.2f}')