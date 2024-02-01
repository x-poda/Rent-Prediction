import os
import pickle
import optuna
import pandas as pd
import xgboost as xgb
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

LOCAL_PATH = "C:/Users/DELL/PycharmProjects/Zoi Task/data/"
IMAGES_PATH = "C:/Users/DELL/PycharmProjects/Zoi Task/img/"
FILENAME = "rent_data_clean.csv"


class XGBoostRegressorTrainer:
    def __init__(self, enable_categorical=True, experiment_name=None, ):
        self.structural_features = ['garden',
                                    'noRooms',
                                    'lift',
                                    'livingSpace',
                                    'cellar',
                                    'hasKitchen',
                                    'newlyConst',
                                    'balcony',
                                    'floor',
                                    'yearConstructedRange',
                                    'condition',
                                    'total_rent_new',
                                    'typeOfFlat_loft',
                                    'typeOfFlat_maisonette',
                                    'typeOfFlat_non_luxury_type',
                                    'typeOfFlat_penthouse',
                                    'typeOfFlat_terraced_flat'
                                    ]
        self.text_features = ["descr_summary_embeddings", "classification_rate", "descr_word_count"]

        self.target = 'total_rent_new'
        self.use_text_features = False
        self.enable_categorical = enable_categorical
        self.experiment_name = experiment_name
        self.hyperparameter_space = None
        self.rent_data = None
        self.model = None

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.dtrain_reg = None
        self.dvalid = None
        self.dtest_reg = None

    def data_loading_and_prep(self):
        self.rent_data = pd.read_csv(os.path.join(LOCAL_PATH, FILENAME))

        # if self.use_text_features:
        #     features_for_prediction = self.structural_features + self.text_features
        # else:
        #     features_for_prediction = self.structural_features
        #
        # self.rent_data = self.rent_data.loc[:, features_for_prediction]

        # Additional data preprocessing or cleaning steps if needed
        for col in ["condition"]:
            self.rent_data[col] = self.rent_data[col].astype('category')
        X, y = self.rent_data.drop('total_rent_new', axis=1), self.rent_data[['total_rent_new']]

        # Split the data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, random_state=1)

        # Create regression matrices
        self.dtrain_reg = xgb.DMatrix(self.x_train, self.y_train, enable_categorical=self.enable_categorical)
        self.dtest_reg = xgb.DMatrix(self.x_test, self.y_test, enable_categorical=self.enable_categorical)

    def objective(self, trial):
        self.hyperparameter_space = {
            "objective": "reg:squarederror",
            "verbosity": 0,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }

        model_with_optuna = xgb.train(
            params=self.hyperparameter_space,
            dtrain=self.dtrain_reg,
            num_boost_round=100,
        )
        predictions_with_optuna = model_with_optuna.predict(self.dtest_reg)
        # Calculating metrics for evaluation, using rmse as the objective to minimize
        rmse = mean_squared_error(self.y_test, predictions_with_optuna, squared=False)
        return rmse

    def train_model(self):
        study = optuna.create_study(direction='minimize', study_name='regression')
        study.optimize(self.objective, n_trials=30)

        # This should be logged in mlflow
        print('Best hyperparameters:', study.best_params)
        print('Best RMSE:', study.best_value)

        self.model = xgb.train(
            params=study.best_params,
            dtrain=self.dtrain_reg,
            num_boost_round=100
        )

    def evaluate_regressor(self):
        # generate predictions using test set.
        predictions = self.model.predict(self.dtest_reg)
        final_rmse = mean_squared_error(self.y_test, predictions, squared=False)
        final_r2 = r2_score(self.y_test, predictions)
        print("Final RMSE", final_rmse)
        print("Final R2", final_r2)

    def plot_regression_results(self):
        predictions = self.model.predict(self.dtest_reg)

        # Plotting Feature Importance and saving it locally
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 8))
        xgb.plot_importance(self.model, importance_type='weight', title='importance_type=weight',
                            max_num_features=10, show_values=False, ax=ax1, )
        xgb.plot_importance(self.model, importance_type='cover', title='importance_type=cover',
                            max_num_features=10, show_values=False, ax=ax2)
        xgb.plot_importance(self.model, importance_type='gain', title='importance_type=gain',
                            max_num_features=10, show_values=False, ax=ax3)
        plt.tight_layout()
        if self.use_text_features:
            importance_fig_path = os.path.join(IMAGES_PATH, "feature_importance_with_text.png")
        else:
            importance_fig_path = os.path.join(IMAGES_PATH, "feature_importance.png")
        plt.savefig(importance_fig_path, bbox_inches='tight')
        plt.close()

        # Plotting True vs Predicted Values and saving it locally
        plt.figure(figsize=(10, 5))
        plt.scatter(self.y_test, predictions)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], color='red',
                 linewidth=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values')
        if self.use_text_features:
            true_vs_pred_fig_path = os.path.join(IMAGES_PATH, "true_vs_predicted_values_with_text.png")
        else:
            true_vs_pred_fig_path = os.path.join(IMAGES_PATH, "true_vs_predicted_values.png")

        plt.savefig(true_vs_pred_fig_path, bbox_inches='tight')
        plt.close()

        return {
            'feature_importance': importance_fig_path,
            'true_vs_predicted': true_vs_pred_fig_path,
        }

    def save_model(self):
        # Create the "models" folder if it doesn't exist
        model_folder = "models"
        os.makedirs(model_folder, exist_ok=True)

        # Get the current timestamp using datetime module
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Construct the filename with the timestamp and folder path
        if self.use_text_features:
            filename = os.path.join(model_folder, f"xgbregressor_model_with_text_{timestamp}.pkl")
        else:
            filename = os.path.join(model_folder, f"xgbregressor_model_{timestamp}.pkl")

        # Save the model inside the "models" folder
        with open(filename, "wb") as file:
            pickle.dump(self.model, file)


if __name__ == "__main__":
    xgboost_regressor = XGBoostRegressorTrainer()
    xgboost_regressor.data_loading_and_prep()
    xgboost_regressor.train_model()
    xgboost_regressor.evaluate_regressor()
    xgboost_regressor.plot_regression_results()
    xgboost_regressor.save_model()
