import numpy as np
import pandas as pd
import xgboost as xgb
import os
import cv2
import shap
from scipy.stats import randint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import scikitplot as skplt
from xgboost import plot_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from skimage import filters
from scipy import ndimage
from skimage import morphology
from skimage import measure
from skimage import exposure
from cellpose import models
from catboost import CatBoostClassifier
from scipy.spatial.distance import cdist

def compute_small_cell_labels(image):

    # Enhance the intensity of the cells using histogram equalization
    equalized_image = exposure.equalize_hist(image)
    
    # Segment the cells in the enhanced image using the Cellpose method
    model = models.Cellpose(gpu=False, model_type='cyto')
    
    # Run the model on the equalized image
    masks, flows, styles, diams = model.eval(equalized_image, diameter=None, channels=[0,0])

    return masks

def get_shap(model, X_test):
    # Load JS visualization code to notebook
    shap.initjs()

    # Create explainer
    explainer = shap.TreeExplainer(model)

    # Get Shapley values
    shap_values = explainer.shap_values(X_test)

    return shap_values


def compute_big_cell_labels(image):
    
    plt.figure(figsize=(10, 10))
    plt.title('0. Image')
    plt.imshow(image, cmap='gray')

    # Step 1: Median filter
    denoised = ndimage.median_filter(image, size=3)
    plt.figure(figsize=(10, 10))
    plt.subplot(321)
    plt.title('1. Denoised Image')
    plt.imshow(denoised, cmap='gray')

    # Step 2: Thresholding
    _, li_thresholded = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    plt.subplot(322)
    plt.title('2. Thresholded Image')
    plt.imshow(li_thresholded, cmap='gray')

    # Step 3: Invert Threshold
    li_thresholded = np.array([[not cell for cell in row] for row in li_thresholded])
    plt.subplot(323)
    plt.title('3. Inverted Threshold')
    plt.imshow(li_thresholded, cmap='gray')

    # Step 4: Fill holes
    filled_holes = ndimage.binary_fill_holes(li_thresholded).astype(bool)
    plt.subplot(324)
    plt.title('4. Filled Holes')
    plt.imshow(filled_holes, cmap='gray')

    # Step 5: Remove small holes
    width = 10
    remove_holes = morphology.remove_small_holes(filled_holes, width ** 3)
    plt.subplot(325)
    plt.title('5. Removed Small Holes')
    plt.imshow(remove_holes, cmap='gray')

    # Step 6: Remove small objects
    remove_objects = morphology.remove_small_objects(remove_holes, width ** 3)
    plt.subplot(326)
    plt.title('6. Removed Small Objects')
    plt.imshow(remove_objects, cmap='gray')

    plt.tight_layout()
    plt.show()

    # Step 7: Label the objects
    labels = measure.label(remove_objects)
    plt.figure(figsize=(5, 5))
    plt.title('7. Labeled Image')
    plt.imshow(labels, cmap='nipy_spectral')

    plt.show()
    return labels

def plot_features_importance(classifier):
    # Create a new figure with the desired size
    fig, ax = plt.subplots(figsize=(20, 30))

    # Plot the feature importance on that figure
    plot_importance(classifier, ax=ax)
    plt.show()

def get_hard_disk_path(type):
    # List of potential paths
    if type == "Segmentation":
        paths = [
            "D:/data_for_seg/",
            "E:/data_for_seg/"
        ]
    elif type == "DL_augmented":
        paths = [
            "D:/data_for_DL_augmented/",
            "E:/data_for_DL_augmented/"
        ]
    elif type == "DL":
        paths = [
            "D:/data_for_DL/",
            "E:/data_for_DL/"
        ]

    actual_path = None
    for path in paths:
        if os.path.exists(path):
            actual_path = path
            print(f"Successfully loaded data from {path}")
    
    return actual_path

def show_samples(image_dir, samples, title):
    num_misclassified = len(samples)
    
    # Define the number of rows and columns for subplots
    num_rows = 2
    num_cols = num_misclassified  # +1 to ensure at least 2 columns
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 6))
    
    for i, misclassified_sample in enumerate(samples):
        folder_name, sample_name = misclassified_sample.split(' - ')

        # Load the original image
        image_path = os.path.join(image_dir, "z_projection", folder_name, sample_name)
        image = cv2.imread(image_path)
        if num_misclassified == 1:
            ax = axs[0]
        else:
            ax = axs[0, i % num_cols]  # Use modulo to handle multiple rows
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(sample_name)
        ax.axis('off')

        # Load the labeled image (if named as "labeled_sample_name")
        labeled_sample_name = "labeled_" + sample_name
        image_path_labeled = os.path.join(image_dir, "labeled", folder_name, labeled_sample_name)
        labeled_image = cv2.imread(image_path_labeled)
        if num_misclassified == 1:
            ax = axs[1]
        else:
            ax = axs[1, i % num_cols]  # Use modulo to handle multiple rows
        ax.imshow(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB))
        ax.set_title(labeled_sample_name)
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def visualize_correlation(df):
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation')
    plt.show()

def load_data(path, csv_name):
    data = pd.read_csv(path + csv_name)
    X = data.iloc[:, 2:]
    y = data.iloc[:, 1]
    unique_labels = y.unique()
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    y = y.map(label_mapping)
    return X, y, unique_labels, label_mapping, data


def train_model(X_train, y_train, unique_labels, model, random_state=42):
    if model == "RandomForest":
        best_params = {'max_depth': 6, 'min_samples_leaf': 2, 'min_samples_split': 7, 'n_estimators': 65}
        classifier = RandomForestClassifier(**best_params, random_state=random_state)
    elif model == "Xgboost":
        best_params = {'colsample_bytree': 0.3064379361316407, 'learning_rate': 0.030294308573206426, 'max_depth': 5, 'n_estimators': 252}
        classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=len(unique_labels), **best_params, random_state=random_state)
    elif model == "Catboost":
        best_params = {"learning_rate": 0.03514033339475594, "depth": 4, "subsample": 0.6983280497870886, "colsample_bylevel": 0.4327452011540816, "min_data_in_leaf": 12}
        classifier = CatBoostClassifier(iterations=1000, bootstrap_type="Bernoulli", verbose=False, **best_params, random_state=random_state)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    classifier.fit(X_train, y_train, sample_weight=sample_weights)
    return classifier


def plot_confusion_matrix(y_test, y_pred, unique_labels):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=unique_labels,
                yticklabels=unique_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def get_classification_details(y_test, y_pred, data):
    wellclassified_indices = np.where(y_test == y_pred)[0]
    wellclassified_samples = data.iloc[y_test.index[wellclassified_indices], 1] + ' - ' + data.iloc[
        y_test.index[wellclassified_indices], 0]
    misclassified_indices = np.where(y_test != y_pred)[0]
    misclassified_samples = data.iloc[y_test.index[misclassified_indices], 1] + ' - ' + data.iloc[
        y_test.index[misclassified_indices], 0]
    return wellclassified_samples, misclassified_samples


def plot_average_f1_scores(ranked_labels, average_f1_scores):
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(ranked_labels)), [average_f1_scores[label] for label in ranked_labels])
    
    # Compute the overall average F1-score
    overall_average_f1 = sum(average_f1_scores.values()) / len(average_f1_scores)
    
    # Add an horizontal line for the overall average F1-score
    ax.axhline(overall_average_f1, color='red', linestyle='--', alpha=0.7)
    
    # Set the title which now includes the overall average F1-score
    ax.set_title(f'Class Ranking Based on Average F1-score\nOverall Avg. F1-score: {overall_average_f1:.2f}', fontsize=16, color='red', y=1.08)
    
    ax.set_xticks(range(len(ranked_labels)))
    ax.set_xticklabels(ranked_labels, rotation=45, fontsize=12)
    ax.set_ylabel('Average F1-score', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()



def run_randomsearch_random_forest(X, y, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    classifier = RandomForestClassifier(random_state=random_state)
    param_dist = {
        'n_estimators': randint(20, 200),  # Sample integer values between 20 and 200
        'max_depth': randint(1, 10),        # Sample integer values between 1 and 10
        'min_samples_split': randint(2, 10),  # Sample integer values between 2 and 10
        'min_samples_leaf': randint(1, 5)    # Sample integer values between 1 and 5
    }
    
    strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    random_search = RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_dist,
        n_iter=100,  # Number of parameter settings to try
        scoring='roc_auc',
        cv=strat_kfold,
        n_jobs=-1,
        verbose=3,
        random_state=random_state
    )
    
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    random_search.fit(X_train, y_train, sample_weight=sample_weights)
    
    return random_search, X_test, y_test

def evaluate_gridsearch(grid_search, X_test, y_test, unique_labels):
    eval_auc = roc_auc_score(y_test, grid_search.best_estimator_.predict_proba(X_test), multi_class='ovr')
    y_pred = grid_search.best_estimator_.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, unique_labels)
    y_probas = grid_search.best_estimator_.predict_proba(X_test)
    skplt.metrics.plot_roc(y_test, y_probas, figsize=(8, 6), title="ROC Curves")
    plt.show()
    return eval_auc, grid_search.best_params_


def calculate_closest_boundary_distances(labels):
    try:
        regions = measure.regionprops(labels)
        min_distances = []

        for i, region1 in enumerate(regions):
            min_distance = np.inf
            boundary1 = np.array(region1.coords)

            for j, region2 in enumerate(regions):
                if i != j:
                    boundary2 = np.array(region2.coords)

                    # Compute all pairwise distances between boundary points
                    distance_matrix = cdist(boundary1, boundary2)

                    # Find the minimum distance to this region
                    current_min_distance = np.min(distance_matrix)
                    if current_min_distance < min_distance:
                        min_distance = current_min_distance

            if min_distance != np.inf:
                min_distances.append(min_distance)

        average_min_distance = np.mean(min_distances)

        return average_min_distance

    except:
        return None
