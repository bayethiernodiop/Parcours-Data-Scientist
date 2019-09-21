

#-----------------------------------------------------------------------
def downcast_dtypes(dataframe, save=False):
    '''This function tries to downcast integer and floating dtypes columns
    to the smallest numerical corresponding dtype.
    It returns a dictionnary of the actually downcasted dtypes.'''
    
    import pandas as pd
    
    # initialise the dict of downcasted dtypes for features
    dict_dtypes = {}
    
    # getting list of integer columns
    columns_int = dataframe.select_dtypes(include=['integer']).columns
    
    for column in columns_int:
        old_dtype = str(dataframe[column].dtypes)
        # trying to downcast integer columns (np.int8 to np.int64)
        dataframe[column] = pd.to_numeric(dataframe[column], downcast='integer')
        new_dtype = str(dataframe[column].dtypes)
        
        # if dtype was downcasted
        if new_dtype != old_dtype:
            print("Column {} downcasted from {} to {}.".format(column, old_dtype, new_dtype))
            # add new key in dictionnary
            dict_dtypes[column] = str(dataframe[column].dtypes)

    # getting list of floatting columns
    columns_float = dataframe.select_dtypes(include=['floating']).columns
    
    for column in columns_float:
        old_dtype = str(dataframe[column].dtypes)
        # trying to downcast float columns (np.float32 to np.float64)
        dataframe[column] = pd.to_numeric(dataframe[column], downcast='float')
        new_dtype = str(dataframe[column].dtypes)
        
        # if dtype was downcasted
        if new_dtype != old_dtype:
            print("Column {} downcasted from {} to {}.".format(column, old_dtype, new_dtype))
            # add new key in dictionnary
            dict_dtypes[column] = str(dataframe[column].dtypes)
        
    # Saving as a json file
    if save==True:
        import json
        
        filename = 'dict_dtypes.json'
        with open(filename, 'w') as outfile:
            json.dump(dict_dtypes, outfile)
        
    # return dict of downcasted dtypes
    return dict_dtypes

#-----------------------------------------------------------------------
def global_filling_rate(dataframe):
    """Compute and displays global filling rate of a DataFrame"""

    # get the numbers of rows and columns in the dataframe
    nb_rows, nb_columns = dataframe.shape
    print("DataFrame has {} rows and {} columns.".format(nb_rows, nb_columns))

    # get the number of non-Nan data in the dataframe
    nb_data = dataframe.count().sum()

    # computing the filling rate
    filling_rate = nb_data / (nb_rows * nb_columns)
    missing_rate = 1 - filling_rate

    # computing the total missing values
    missing_values = (nb_rows * nb_columns) - nb_data

    # display global results
    print("")
    print("Global filling rate of the DataFrame: {:.2%}".format(filling_rate))
    print("Missing values in the DataFrame: {} ({:.2%})"
          .format(missing_values, missing_rate))

    # compute number of rows with missing values
    mask = dataframe.isnull().any(axis=1)
    rows_w_missing_values = len(dataframe[mask])
    rows_w_missing_values_percentage = rows_w_missing_values / nb_rows

    # display results
    print("")
    print("Number of rows with missing values: {} ({:.2%})"
          .format(rows_w_missing_values, rows_w_missing_values_percentage))

    # compute number of columns with missing values
    mask = dataframe.isnull().any(axis=0)
    cols_w_missing_values = len(dataframe[dataframe.columns[mask]].columns)
    cols_w_missing_values_percentage = cols_w_missing_values / nb_columns

    # display results
    print("Number of columns with missing values: {} ({:.2%})"
          .format(cols_w_missing_values, cols_w_missing_values_percentage))

    
#-----------------------------------------------------------------------
def columns_filling_rate(dataframe, columns='all', missing_only=False):
    """Calculate and displays the filling rate for
    a particular column in a pd.DataFrame."""
    
    # Importations
    import pandas as pd
    import numpy as np
    
    # If 'feature' is not specified
    if columns == 'all':
        columns = dataframe.columns
        
    
    # initialization of the results DataFrame
    results = pd.DataFrame(columns=['nb_values', 'missing_values', 'filling_rate'])
        
    # for each feature
    for column in columns:

        # Count of the values on each column
        values_count = dataframe[column].count()
        
        # Computing missing values
        nb_rows = dataframe.shape[0]
        missing_values = nb_rows - values_count

        # Computing filling rates
        filling_rate = values_count / nb_rows
        if missing_only and filling_rate == 1:
            filling_rate = np.nan
        
        # Adding a row in the results' dataframe
        results.loc[column] = [values_count, missing_values, filling_rate]

    # Sorting the features by number of missing_values
    results = results.dropna(subset=['filling_rate'])
    results = results.sort_values('filling_rate')
    
    return results

#-----------------------------------------------------------------------
def categorical_matrix(dataframe, theil_u=True, return_results=False):
    """Displays a kind of "correlation matrix" including categorical features."""

    # loading library
    from dython.nominal import associations

    # Get the categorical and boolean columns
    categorical_columns = list(dataframe.select_dtypes(include='category').columns)
    categorical_columns += list(dataframe.select_dtypes(include='bool').columns)

    # Drop NaN values to avoid errors
    df_for_correlations = dataframe.dropna()
    
    # Drop 'object', 'datetime' and 'timedelta' columns
    df_for_correlations = df_for_correlations.select_dtypes(exclude=['object', 'datetime', 'timedelta'])

    # Calculate associations (returns None) and display graph
    corr = associations(
        df_for_correlations,
        figsize=(15,7),
        theil_u=theil_u, # asymetric measure of correlation for nominal feature
        nominal_columns=categorical_columns,
        return_results=return_results
    )
    
    # Returns correlation matrix, if requested
    if return_results:
        return corr

#-----------------------------------------------------------------------
def smart_imputation(dataframe):
    """Do column-wise imputation based on the dtypes."""
    
    # Load libraries
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.compose import ColumnTransformer
    
    # Getting data types of features
    categorical_features = list(dataframe.select_dtypes(include=['category']).columns)
    numerical_features = list(dataframe.select_dtypes(include='number').columns)
    
    #------------------------------------------
    # Simple imputation for categorical features
    categorical_imputer = SimpleImputer(
        strategy='constant',
        fill_value='missing',
    )
        
    # Iterative imputation for numerical features
    numerical_imputer = IterativeImputer(
        max_iter=10,
        # random_state=42,
        add_indicator=False,
    )

    # Column-wise differentiate imputation for numeric and categorical features
    smart_imputer = ColumnTransformer(transformers=[
            ('categorical_imputer', categorical_imputer, categorical_features),
            ('numerical_imputer', numerical_imputer, numerical_features)
    ])
    
    #------------------------------------------
    # Proceed to imputation (returns a np.array)
    array_imputed = smart_imputer.fit_transform(dataframe)
    
    # Convert back to a pd.DataFrame
    name_columns_imputed = categorical_features + numerical_features
    columns_imputed = pd.DataFrame(array_imputed, columns=name_columns_imputed)
    
    # Getting the non-imputed colums
    columns_not_imputed = dataframe.select_dtypes(exclude=['number', 'bool', 'category'])
    
    # Concatenating imputed and non-imputed columns
    df_imputed = pd.concat([columns_imputed, columns_not_imputed], sort=True, axis=1)
    
    # Getting dtypes of numerical features in the original dataframe
    dtypes_dict = dict(dataframe[numerical_features].dtypes)
    
    # Change dtypes to fit the original dtypes
    df_imputed[categorical_features] = df_imputed[categorical_features].astype(dtype='category')
    df_imputed[numerical_features] = df_imputed[numerical_features].astype(dtype=dtypes_dict)
    
   
    # Return a dataframe with imputed and non-imputed columns
    return df_imputed

#-----------------------------------------------------------------------
def find_colinear_features(corrs, threshold=0.8, target_name='TARGET'):
    """Find colinear (highly correlated) features in a correlation DataFrame.
    This correlation DataFrame can be get from:
        - df.corr() for numerical features
        - dython.nominal.associations for categorical or mixed features
        
    Doesn't handle asymetry of Theil U coefficient yet.
    """
    
    # Initialize a dictionnary for colinear variables
    colinear_dict = {}

    # Masking target columns
    #-----------------------
    # In case a a single target column (or not specified)
    try:
        mask_target = (corrs.columns != target_name)
        
    # In case a several target columns
    except :
        mask_target = (~corrs.columns.isin(target_name))

        
    # Getting subsets of highly correlated features
    #----------------------------------------------
    # Initialize the list of substets
    list_of_correlated_subsets = []
    
    # For each column, record the variables that are above the threshold
    for feature in corrs:        
        # Select features with correlation coefficient higher than threshold
        mask1 = (corrs[feature].abs() > threshold)
        # Avoid selecting self feature (auto-correlation == 1)
        mask2 = (corrs.loc[feature].index != feature)
        # Final mask
        mask = mask1 & mask2
        # Create the subset of correlated features
        subset_of_correlated_features = set(corrs.index[mask])

        # If the subset of correlated features is not empty
        if subset_of_correlated_features:
            # Add the feature to the set
            subset_of_correlated_features.add(feature)
            # Add the values to the list of subsets, if not already
            if subset_of_correlated_features not in list_of_correlated_subsets:
                list_of_correlated_subsets.append(subset_of_correlated_features)

    # Checking for "transitivity" of correlation relationships
    #---------------------------------------------------------
    # for subset in list_of_correlated_subsets:
        # if 
    
    
    # Display and return results

    
    if not list_of_correlated_subsets:
        print("No found correlated features above threshold {}.".format(threshold))
    else:
        print("Subset(s) of correlated features above threshold {}:".format(threshold))
        for subset in list_of_correlated_subsets:
            print(subset)
    
    # Return results
    return list_of_correlated_subsets


#-----------------------------------------------------------------------
def PCA_features_reduction(X_std, var_threshold=0.9):
    """Return the principal components from PCA, until variance threshold."""

    from sklearn import decomposition
    
    # 
    print("Initial number of features:", X_std.shape[1])
    
    # Processing the PCA
    pca = decomposition.PCA()
    pca.fit(X_std)
    
    # Getting the explained variance ratio for each principal component
    scree = pca.explained_variance_ratio_

    # Getting the number of principal components to reach variance thresholds
    mask = scree.cumsum() > var_threshold
    nb_selected_features = len(scree[~mask]) + 1
    print("Number of selected features:", nb_selected_features)
    
    # Compute and displays the actual ratio of explained variance
    explained_variance_sum = scree.cumsum()[nb_selected_features-1]
    print("Cumulative explained variance:  {:.2f}%".format(explained_variance_sum*100))
    
    # Getting the projection of the data on the first components
    X_projected = pca.transform(X_std)[:,:nb_selected_features]
    
    return X_projected


#------------------------------------------------------------------------
def convert_timeseries_to_float(dataframe):
    import datetime
    import numpy as np

    df = dataframe.copy()
    
    # Get 'timedelta' and 'datetime' features
    timedelta_features = list(df.select_dtypes('timedelta').columns)
    datetime_features = list(df.select_dtypes('datetime').columns)

    # Iterating over 'timedelta' features
    for feature in timedelta_features:
        # converting NaT to NaN
        df[feature] = df[feature].fillna(np.nan)
        # filtering off the missing values
        mask = df[feature].notnull()
        # Convert 'timedelta' features to float
        df.loc[mask, feature]  = dataframe[feature][mask]  / datetime.timedelta(days=1)
        # dataframe[feature] = dataframe[feature].total_seconds()
        # Converting dtype of the column
        # df[feature] = df[feature].astype(float)

    # Iterating over 'datetime' features
    for feature in datetime_features:
        # converting NaT to NaN
        df[feature] = df[feature].fillna(np.nan)
        # filtering off the missing values
        mask = df[feature].notnull() # pd.Series of boolean
        # Convert 'datetime' values to float
        df.loc[mask, feature] = dataframe[feature][mask].map(lambda x: x.timestamp())
        # Converting dtype of the column
        # df[feature] = df[feature].astype(float)

    return df

#------------------------------------------------------------------------
def get_feature_names(columnTransformer):
    """This function returns features names from a 
    ColumnTransformer object, to keep track of featurse names."""

    output_features = []

    for transformers in columnTransformer.transformers:

        if transformers[0]!='remainder':
            pipeline = transformers[1]
            features = transformers[2]

            for i in pipeline:
                trans_features = []
                if hasattr(i,'categories_'):
                    for feature, categories in zip(features,i.categories_):
                        trans_features.extend(['{}_{}'.format(feature,cat)
                                                   for cat in categories])
                else:
                    trans_features = features
            output_features.extend(trans_features)

    return output_features


#----------------------------------------------------------------------
def define_preprocessor(dataframe, supervised=False):
    """This function returns a 'preprocessor' object wrapping
    different preprocessing column-wise steps.
        - impute missing values
        - target encode categorical features
        - standardization of all features
    """

    # Load required libraries and modules
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from category_encoders.target_encoder import TargetEncoder
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.preprocessing import FunctionTransformer
    import datetime
    
    # Warning if some columns have 'timedelta' or 'datetime' dtypes
    if not dataframe.select_dtypes(include=['timedelta', 'datetime']).empty:
        import warnings
        warnings.warn("Convert columns with 'timedelta' and 'datetime' dtypes before preprocessing.", UserWarning)
           
    # Warning if some columns' have object dtypes (not handled)
    if not dataframe.select_dtypes(include=['object']).empty:
        import warnings
        warnings.warn("Columns with 'object' dtypes are not handled and returned inchanged.", UserWarning)
        
   
    
    # Preprocessing pipeline for categorical features (supervised case)
    if supervised:
        categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # simple imputation 
                ('target_encoder', TargetEncoder()), # target encoding
                ('scaler', StandardScaler()), # standardization after target encoding
                ])
    
    # Preprocessing pipeline for categorical features (unsupervised case)
    else:
        categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # simple imputation 
                ('one_hot_encoder', LabelBinarizer()), # one_hot encoding for categorical nominal features
                ('scaler', StandardScaler()), # standardization after target encoding
                ])
    
    # Preprocessing pipeline for numeric features
    numeric_transformer = Pipeline(steps=[
            ('imputer', IterativeImputer(max_iter=10)), # iterative imputation
            ('scaler', StandardScaler()), # standardization
             ])
    
    # Preprocessing pipeline for other dtypes features
    other_transformer = Pipeline(steps=[
            ('identity', FunctionTransformer(validate=False)), # identity function
             ])

    # Column-wise preprocessor using pipelines for numeric and categorical features
    preprocessor = ColumnTransformer(transformers=[
            ('cat', categorical_transformer, list(dataframe.select_dtypes(include=['category', 'bool']).columns)),
            ('num', numeric_transformer, list(dataframe.select_dtypes(include='number', exclude=['datetime', 'timedelta']).columns)),
            ('obj', other_transformer, list(dataframe.select_dtypes(include=['object', 'datetime', 'timedelta']).columns)),
            ])
    
    return preprocessor


#------------------------------------------------------------------------
def preprocessing(dataframe, target_name='TARGET', supervised=False):
    """Apply the preprocessing and get back columns name to
    returns a pd.DataFrame."""
    
    import pandas as pd
    
    #-------------------------------------------------------------------
    # Function to get the features names from a ColumnTransformer object
    def get_feature_names(columnTransformer):

        output_features = []

        for transformers in columnTransformer.transformers:

            if transformers[0]!='remainder':
                pipeline = transformers[1]
                features = transformers[2]

                for i in pipeline:
                    trans_features = []
                    if hasattr(i,'categories_'):
                        for feature, categories in zip(features,i.categories_):
                            trans_features.extend(['{}_{}'.format(feature,cat)
                                                       for cat in categories])
                    else:
                        trans_features = features
                output_features.extend(trans_features)

        return output_features
    
    #------------------------------------------------------------------
    # Function to revert a np.ndarray to pd.DataFrame with columns names
    def revert_to_df(X_preprocessed, preprocessor):
        features_name = get_feature_names(preprocessor)
        X_preprocessed = pd.DataFrame(X_preprocessed, columns=features_name)
        return X_preprocessed
        
    #------------------------------------------------------------------
    X = dataframe.copy()
    
    # Specific operations for supervised case
    if supervised:
        # Dropping rows with missing target values
        X = X.dropna(subset=[target_name])

        # Setting the target variable (used for coloration only)
        y = X[target_name]

        # Removing the target from features
        X = X.drop(columns=[target_name])


    # Instanciate preprocessor
    preprocessor = define_preprocessor(X, supervised=supervised)
    
    # Apply the preprocessor to the data
    if supervised:
        X_preprocessed = preprocessor.fit_transform(X, y)
        X_preprocessed = revert_to_df(X_preprocessed, preprocessor)
        return (X_preprocessed, y_preprocessed)
    
    else:
        X_preprocessed = preprocessor.fit_transform(X)
        X_preprocessed = revert_to_df(X_preprocessed, preprocessor)
        return X_preprocessed
    
    
#-----------------------------------------------------------
def sampling_from_ndarray(array, size, replace=False):
    """Sample from np.ndarray. Do NOT work with pd.DataFrame"""
    idx = np.random.choice(array.shape[0], size, replace=replace)
    sample = array[idx,:]
    return sample


#-----------------------------------------------------------
def display_scree_plot(X_std):
    """This function displays the scree plot of proper values
    for PCA decomposition.
    
    Parameters
    --------
    X_std (pandas.DataFrame or np.ndarray):
        The standardised data (features).
   
    Future improvements
    -------------------
    * english translation
    * unit thicks
    """
    
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    
    pca = PCA()
    pca.fit(X_std)
    scree = pca.explained_variance_ratio_*100
    
    plt.figure(figsize=(20,10))
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres", fontsize=15)
    plt.show(block=False)

    
#-----------------------------------------------------------
def collapse_categories_by_frequency(categorical_Series, threshold=0.05, replacement='other'):
    """This function collapse together categories with frequencies
    (or relative frequencies) under the threshold. It returns a pd.Series
    with these categories replaced by 'replacement'."""

    # get frequencies of each modality
    categorical_Series.value_counts()

    # case where threshold is a frequency
    if threshold > 1:
        # define mask for modalities under threshold
        mask = (categorical_Series.value_counts() < threshold)

    # case where threshold is a relative frequency
    if threshold > 0 and threshold <1:
        # compute total frequency
        total_frequency = len(categorical_Series)
        # define mask for modalities under threshold
        mask = (categorical_Series.value_counts() < threshold*total_frequency)

    # apply mask and get categories names
    categories_to_collapse = list(categorical_Series.value_counts()[mask].index)

    # remap all 'categories to collapse' to the replacement value
    result = categorical_Series.apply(lambda x: replacement if x in categories_to_collapse else x)
    
    # displays informations
    print("Number of categorical modalities before collapsing: {}".format(categorical_Series.nunique()))
    print("Number of categorical modalities after collapsing: {}".format(result.nunique()))

    # return transformed pd.Series
    return result

#-----------------------------------------------------------------------
def IQR_outliers_mask(Series):
    """This function return a mask of outliers.
    Outliers are those with value less than (Q1 + 1.5*IQR) or bigger than (Q3 + 1.5*IQR)"""
    
    Q1 = Series.quantile(q=0.25, interpolation='linear')   # first quartile
    Q3 = Series.quantile(q=0.75, interpolation='linear')   # third quartile
    IQR = Q3 - Q1                                                      # inter-quartile range
    
    # definition of mask of outliers
    mask = (Series < Q1 - 1.5*IQR) | (Series > Q3 + 1.5*IQR)
    
    # application of mask
    outliers = Series[mask]
    
    # Displays number of outliers
    print("Number of detected outliers:", len(mask[mask]))
    
    # return the mask of outliers
    return mask