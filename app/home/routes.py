# -*- encoding: utf-8 -*-

from app.home import blueprint
from flask import render_template, redirect, url_for, request
from flask_login import login_required, current_user
from app import login_manager
from jinja2 import TemplateNotFound
from flask import make_response, send_file
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import models as algorithms
import plotfunctions as plotfun
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from app.base.models import User, Industry, NetExports
import joblib
import seaborn as sns
from matplotlib import pyplot as plt 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
from tensorflow import keras
from keras.optimizers import RMSprop
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_squared_log_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
scaler = MinMaxScaler(feature_range=(0, 1))

@blueprint.route('/index')
@login_required
def index():
    users = User.query.count()
    activities = Industry.query.count()
    return render_template('index.html', segment='index', users=users, activities=activities)

@blueprint.route('/analysis', methods = ['GET', 'POST'])
@login_required
def analysis():
    datasets,_,folders = datasetList()
    originalds = []
    featuresds = []
    for i in range(len(datasets)):
        if folders[i] == 'datasets': originalds += [datasets[i]]
        else: featuresds += [datasets[i]]
    if request.method == 'POST':
            f = request.files['file']
            f.save(os.path.join('datasets', f.filename))
            return redirect('/analysis')
    return render_template('analysis/index.html', originalds = originalds, featuresds = featuresds)


@blueprint.route('/datasets/<dataset>')
@login_required
def dataset(description = None, head = None, dataset = None):
    df = loadDataset(dataset)
    print(df)
    try:
        description = df.describe().round(2)
        head = df.head(5)
    except: pass
    return render_template('analysis/dataset.html',
                           description = description.to_html(classes='table table-striped table-hover'),
                           head = head.to_html(index=False, classes='table table-striped table-hover'),
                           dataset = dataset)

@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith( '.html' ):
            template += '.html'

        # Detect the current page
        segment = get_segment( request )

        # Serve the file (if exists) from app/templates/FILE.html
        return render_template( template, segment=segment )

    except TemplateNotFound:
        return render_template('page-404.html'), 404
    
    except:
        return render_template('page-500.html'), 500

# Helper - Extract current page name from request 
def get_segment( request ): 

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment    

    except:
        return None  

def datasetList():
    datasets = [x.split('.')[0] for f in ['datasets', 'preprocessed'] for x in os.listdir(f)]
    extensions = [x.split('.')[1] for f in ['datasets', 'preprocessed'] for x in os.listdir(f)]
    folders = [f for f in ['datasets', 'preprocessed'] for x in os.listdir(f)]
    return datasets, extensions, folders

#Load columns of the dataset
def loadColumns(dataset):
    datasets, extensions, folders = datasetList()
    if dataset in datasets:
        extension = extensions[datasets.index(dataset)]
        if extension == 'txt':
            df = pd.read_table(os.path.join(folders[datasets.index(dataset)], dataset + '.txt'), nrows=0)
        elif extension == 'csv':
            df = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'), nrows=0)
        elif extension == 'xlsx':
            df = pd.read_excel(os.path.join(folders[datasets.index(dataset)], dataset + '.xlsx'), nrows=0)
            # pd.read_excel('myData.xlsx')

        return df.columns
        

#Load Dataset    
def loadDataset(dataset):
    df = None
    datasets, extensions, folders = datasetList()
    if dataset in datasets:
        extension = extensions[datasets.index(dataset)]
        if extension == 'txt':
            df = pd.read_table(os.path.join(folders[datasets.index(dataset)], dataset + '.txt'))
        elif extension == 'csv':
            df = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'))
        elif extension == 'xlsx':
            df = pd.read_excel(os.path.join(folders[datasets.index(dataset)], dataset + '.xlsx'))
        return df

#Dataset Preprocessing
@blueprint.route('/datasets/<dataset>/preprocessing')
@login_required
def preprocessing(dataset = dataset):
    columns = loadColumns(dataset)
    return render_template('analysis/preprocessing.html', dataset = dataset, columns=columns)

@blueprint.route('/datasets/<dataset>/preprocessed_dataset/', methods=['POST'])
@login_required
def preprocessed_dataset(dataset):
    numFeatures = request.form.get('nfeatures')
    manualFeatures = request.form.getlist('manualfeatures')
    datasetName = request.form.get('newdataset')
    response = request.form.get('response')
    dropsame = request.form.get('dropsame')
    dropna = request.form.get('dropna')
    
    df = loadDataset(dataset)

    if dropna == 'all':
        df = df.dropna(axis=1, how='all')
    elif dropna == 'any':
        df.dropna(axis=1, how='any')
        
    filename = dataset + '_'
    try:
        nf = int(numFeatures)
        from sklearn.feature_selection import SelectKBest, chi2
        X = df.drop(str(response), axis=1)
        y = df[str(response)]
        kbest = SelectKBest(chi2, k=nf).fit(X,y)
        mask = kbest.get_support()
        # List of K best features
        best_features = []
        for bool, feature in zip(mask, list(df.columns)):
            if bool: best_features.append(feature)
        #Reduced Dataset
        df = pd.DataFrame(kbest.transform(X),columns=best_features)
        df.insert(0, str(response), y)
        
        filename += numFeatures + '_' + 'NA' + dropna + '_Same' + dropsame + '.csv'
    
    except:
        df = df[manualFeatures]
        filename += str(datasetName) + '_' + str(response) + '.csv'
    
    if dropsame == 'Yes':
        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        df = df.drop(cols_to_drop, axis=1)
    df.to_csv(os.path.join('preprocessed', filename), index=False)
    return redirect('/datasets/' + filename.split('.')[0])


@blueprint.route('/datasets/<dataset>/graphs')
@login_required
def graphs(dataset = dataset):
    columns = loadColumns(dataset)
    return render_template('analysis/graphs.html', dataset = dataset, columns=columns)

@blueprint.route('/datasets/<dataset>/graphprocess/', methods=['POST'])
@login_required
def graph_process(dataset = dataset):
    histogram = request.form.getlist('histogram')
    boxplotcat = request.form.get('boxplotcat')
    boxplotnum = request.form.get('boxplotnum')
    corr = request.form.getlist('corr')
    corrcat = request.form.get('corrcat')

    if corrcat != '': corr += [corrcat]
    ds = loadDataset(dataset)
    import plotfunctions as plotfun
    figs = {}
    if histogram != [''] and histogram != []:
        figs['Histograms'] = str(plotfun.plot_histsmooth(ds, histogram), 'utf-8')
    if corr != [''] and corr != []:
        figs['Correlations'] = str(plotfun.plot_correlations(ds, corr, corrcat), 'utf-8')
    if boxplotcat != '' and boxplotnum != '':
        figs['Box Plot'] = str(plotfun.plot_boxplot(ds, boxplotcat, boxplotnum), 'utf-8')
    if figs == {}: return redirect('/datasets/' + dataset + '/graphs')
    return render_template('analysis/drawgraphs.html', figs = figs, dataset = dataset)


@blueprint.route('/datasets/<dataset>/models')
@login_required
def models(dataset = dataset):
    columns = loadColumns(dataset)
    clfmodels = algorithms.classificationModels()
    predmodels = algorithms.regressionModels()
    return render_template('models/models.html', dataset = dataset,
                           clfmodels = clfmodels,
                           predmodels = predmodels,
                           columns = columns)

@blueprint.route('/datasets/<dataset>/modelprocess/', methods=['POST'])
@login_required
def model_process(dataset = dataset):
    algscore = request.form.get('model')
    res = request.form.get('response')
    kfold = request.form.get('kfold')
    alg, score = algscore.split('.')
    scaling = request.form.get('scaling')
    variables = request.form.getlist('variables')
    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    df = loadDataset(dataset)
    y = df[str(res)]

    if variables != [] and '' not in variables: df = df[list(set(variables + [res]))]
    X = df.drop(str(res), axis=1)
    try: X = pd.get_dummies(X)
    except: pass
    
    predictors = X.columns
    if len(predictors)>10: pred = str(len(predictors))
    else: pred = ', '.join(predictors)    

    if score == 'Classification':
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
        scoring = ['precision', 'recall', 'f1', 'accuracy', 'roc_auc']
        if scaling == 'Yes':
            clf = algorithms.classificationModels()[alg]
            mod = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        else: 
            mod = algorithms.classificationModels()[alg]
        fig = plotfun.plot_ROC(X.values, y, mod, int(kfold))

    elif score == 'Regression':
        from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
        scoring = ['explained_variance', 'r2', 'mean_squared_error']
        if scaling == 'Yes':
            pr = algorithms.regressionModels()[alg]  
            mod = Pipeline([('scaler', StandardScaler()), ('clf', pr)])
        else: mod = algorithms.regressionModels()[alg]
        fig = plotfun.plot_predVSreal(X, y, mod, int(kfold))
    
    scores = cross_validate(mod, X, y, cv=int(kfold), scoring=scoring)
    for s in scores:
        scores[s] = str(round(np.mean(scores[s]),3))
    return render_template('models/scores.html', scores = scores, dataset = dataset, alg=alg,
                           res = res, kfold = kfold, score = score,
                           predictors = pred, response = str(fig, 'utf-8'))


@blueprint.route('/datasets/<dataset>/predict')
@login_required
def predict(dataset = dataset):
    columns = loadColumns(dataset)
    clfmodels = algorithms.classificationModels()
    predmodels = algorithms.regressionModels()
    return render_template('predictions/predict.html', dataset = dataset,
                           clfmodels = clfmodels,
                           predmodels = predmodels,
                           columns = columns)

@blueprint.route('/datasets/<dataset>/prediction/', methods=['POST'])
@login_required
def predict_process(dataset = dataset):
    algscore = request.form.get('model')
    res = request.form.get('response')
    alg, score = algscore.split('.')
    scaling = request.form.get('scaling')
    df = loadDataset(dataset)
    columns = df.columns
    values = {}
    counter = 0
    for col in columns:
        values[col] = request.form.get(col)
        if values[col] != '' and col != res: counter +=1
    
    if counter == 0: return redirect('/datasets/' + dataset + '/predict')
    
    predictors = {}
    for v in values:
        if values[v] != '':
            try: predictors[v] = [float(values[v])]
            except: predictors[v] = [values[v]]

    from sklearn.preprocessing import StandardScaler
    X = df[list(predictors.keys())]
    Xpred = predictors
    #return str(Xpred)
    Xpred = pd.DataFrame(data=Xpred)
    X = pd.concat([X,Xpred])
    X = pd.get_dummies(X)
    Xpred = X.iloc[[-1]]
    X = X[:-1]
    if scaling == 'Yes':
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
        Xpred = pd.DataFrame(scaler.transform(Xpred), columns = X.columns)
    try:
        X = X.drop(str(res), axis=1)
        Xpred = Xpred.drop(str(res), axis=1)
    except: pass
    #Xpred.reset_index(drop=True, inplace=True)
    #X.reset_index(drop=True, inplace=True)
    y = df[str(res)]
    if score == 'Classification':
            mod = algorithms.classificationModels()[alg]
    elif score == 'Regression':
        mod = algorithms.regressionModels()[alg]
    model = mod.fit(X, y)
    #return pd.DataFrame(Xpred).to_html()
    predictions = {}
    predictions['Prediction'] = model.predict(Xpred)[0]
    predictors.pop(res, None)
    for p in predictors:
        if str(predictors[p][0]).isdigit() is True: predictors[p] = int(predictors[p][0])
        else:
            try: predictors[p] = round(predictors[p][0],2)
            except: predictors[p] = predictors[p][0]
    for p in predictions:
        if str(predictions[p]).isdigit() is True: predictions[p] = int(predictions[p])
        else:
            try: predictions[p] = round(predictions[p],2)
            except: continue
    if len(predictors) > 15: predictors = {'Number of predictors': len(predictors)}
    #return str(predictors) + res + str(predictions) + alg + score
    if score == 'Classification':
        classes = model.classes_
        pred_proba = model.predict_proba(Xpred)
        for i in range(len(classes)):
            predictions['Prob. ' + str(classes[i])] = round(pred_proba[0][i],3)    
    return render_template('predictions/prediction.html', predictions = predictions, response = res,
                           predictors = predictors, algorithm = alg, score = score,
                           dataset = dataset)

@blueprint.route('/primary_sector')
@login_required
def primary_sector():
    # data = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'))
    data = pd.read_excel('industrial_indicators.xlsx', engine='openpyxl')
    df_plot = data[['Years','Agiculture', 'Mining & Quarrying', 'Manufacturing', 'Electricity & Water', 'Construction']]
    return render_template('/sectors/primary.html', data = df_plot.to_html(index=False, classes='table table-striped table-hover'))

@blueprint.route('/secondary_activities')
@login_required
def secondary_activities():
    # data = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'))
    data = pd.read_excel('industrial_indicators.xlsx', engine='openpyxl')
    df_plot = data[['Years','Distributions', 'Transport & Comm', 'Finance & Insurance', 'Public Administration']]
    return render_template('/sectors/secondary.html', data = df_plot.to_html(index=False, classes='table table-striped table-hover'))

@blueprint.route('/supportive_activities')
@login_required
def supportive_activities():
    # data = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'))
    data = pd.read_excel('industrial_indicators.xlsx', engine='openpyxl')
    df_plot = data[['Years','Education', 'Health & Social Work', 'Domestic Services', 'Other Services']]
    return render_template('/sectors/supportive.html', data = df_plot.to_html(index=False, classes='table table-striped table-hover'))

@blueprint.route('/primary_diagram')
@login_required
def primary_diagram():
    data = pd.read_excel('industrial_indicators.xlsx', engine='openpyxl')
    df_plot = data[['Agiculture', 'Mining & Quarrying', 'Manufacturing', 'Electricity & Water', 'Construction']]
    df_plot.plot(kind='bar', figsize=(8, 5), fontsize=10)
    plt.title('CORE ACTIVITIES TRENDS')
    plt.xlabel('Years');
    plt.legend(fontsize=15);
    plt.ylabel('US$m');
    plt.legend(fontsize=15);
    plt.show()
    return redirect('/primary_sector')

@blueprint.route('/secondary_diagram')
@login_required
def secondary_diagram():
    data = pd.read_excel('industrial_indicators.xlsx', engine='openpyxl')
    df_plot = data[['Distributions', 'Transport & Comm', 'Finance & Insurance', 'Public Administration']]
    df_plot.plot(kind='bar', figsize=(8, 5), fontsize=10)
    plt.title('SECONDARY ACTIVITIES TRENDS')
    plt.xlabel('Years');
    plt.legend(fontsize=15);
    plt.ylabel('US$m');
    plt.legend(fontsize=15);
    plt.show()
    return redirect('/secondary_activities')

@blueprint.route('/supportive_diagram')
@login_required
def supportive_diagram():
    data = pd.read_excel('industrial_indicators.xlsx', engine='openpyxl')
    df_plot = data[['Education', 'Health & Social Work', 'Domestic Services', 'Other Services']]
    df_plot.plot(kind='bar', figsize=(10, 5), fontsize=10)
    plt.title('SUPPORTIVE ACTIVITIES TRENDS')
    plt.xlabel('Years');
    plt.legend(fontsize=15);
    plt.ylabel('US$m');
    plt.legend(fontsize=15);
    plt.show()
    return redirect('/supportive_activities')


@blueprint.route('/about')
def about():
    return render_template('about/about.html')

@blueprint.errorhandler(500)
def internal_error(e):
    return render_template('errors/page-500.html')

@blueprint.errorhandler(404)
def page_not_found(e):
    return render_template('errors/page-404.html')

@blueprint.route('/forecasting', methods = ["GET","POST"])
@login_required
def forecasting():
    forecasting_model = keras.models.load_model("models/forecasting_model.h5")
    actual_prediction=0
    year=2021
    if request.method == "POST":
        year = request.form['year']
        data_list = []
        data_list.append(int(request.form['agric']))
        data_list.append(int(request.form['mining']))
        data_list.append(int(request.form['manufacturing']))
        data_list.append(int(request.form['electricity_water']))
        data_list.append(int(request.form['construction']))
        data_list.append(int(request.form['distribution']))
        data_list.append(int(request.form['transport']))
        data_list.append(int(request.form['financial']))
        data_list.append(int(request.form['real_estate']))
        data_list.append(int(request.form['public_administration']))
        data_list.append(int(request.form['education']))
        data_list.append(int(request.form['human_health']))
        data_list.append(int(request.form['domestic_services']))
        data_list.append(int(request.form['other_services']))
        data_list.append(int(request.form['net_tax']))
        data_list.append(int(request.form['fism']))
        data_list.append(int(request.form['net_tax_prod']))

        data_array = np.array(data_list).reshape(1,-1)
        prediction = forecasting_model.predict(data_array)
        prediction_2d = prediction[0]
        prediction_1d = prediction_2d[0]
        actual_prediction = int(prediction_1d)
        print(actual_prediction)
    return render_template('/predictions/by_sector.html',actual_prediction=actual_prediction, year=year )

@blueprint.route('/primary_forecasting',  methods = ["GET","POST"])
@login_required
def primary_forecasting():
    actual_prediction=0
    year=2021
    activity_ = ''
    if request.method == "POST":
        if request.form['activity'] == 'Agriculture':
            forecasting_model = keras.models.load_model("models/agric_model.h5")
        elif request.form['activity'] == 'Mining':
            forecasting_model = keras.models.load_model("models/mining_model.h5")
        elif request.form['activity'] == 'Manufacturing':
            forecasting_model = keras.models.load_model("models/manufacturing_model.h5")
        elif request.form['activity'] == 'Construction':
            forecasting_model = keras.models.load_model("models/construction_model.h5")
        else:
            forecasting_model = keras.models.load_model("models/other_activities_model.h5")
        
        year = request.form['year']
        activity_ = request.form['activity']
        data_value = int(request.form['previous_value'])
        data_value = np.reshape(data_value,(-1,1))
        data_valued = scaler.fit_transform(data_value)
        data_val = np.reshape(data_valued, (data_valued.shape[0], data_valued.shape[1], 1))
        prediction = forecasting_model.predict(data_val)
        prediction_2d = prediction[0]
        prediction_1d = prediction_2d[0]
        actual_prediction = int(prediction_1d * 1000)
        print(actual_prediction)
    return render_template('/predictions/primary_forecasting.html', year=year,actual_prediction=actual_prediction, activity_=activity_)

@blueprint.route('/secondary_forecasting',  methods = ["GET","POST"])
@login_required
def secondary_forecasting():
    actual_prediction=0
    year=2021
    activity_ = ''
    if request.method == "POST":
        forecasting_model = keras.models.load_model("models/other_activities_model.h5")
        
        year = request.form['year']
        activity_ = request.form['activity']
        data_value = int(request.form['previous_value'])
        data_value = np.reshape(data_value,(-1,1))
        data_valued = scaler.fit_transform(data_value)
        data_val = np.reshape(data_valued, (data_valued.shape[0], data_valued.shape[1], 1))
        prediction = forecasting_model.predict(data_val)
        prediction_2d = prediction[0]
        prediction_1d = prediction_2d[0]
        actual_prediction = int(prediction_1d * 1000)
        print(actual_prediction)
    return render_template('/predictions/secondary_forecasting.html', year=year,actual_prediction=actual_prediction, activity_=activity_)

@blueprint.route('/supportive_forecasting',  methods = ["GET","POST"])
@login_required
def supportive_forecasting():
    actual_prediction=0
    year=2021
    activity_ = ''
    if request.method == "POST":
        forecasting_model = keras.models.load_model("models/other_activities_model.h5")
        
        year = request.form['year']
        activity_ = request.form['activity']
        data_value = int(request.form['previous_value'])
        data_value = np.reshape(data_value,(-1,1))
        data_valued = scaler.fit_transform(data_value)
        data_val = np.reshape(data_valued, (data_valued.shape[0], data_valued.shape[1], 1))
        prediction = forecasting_model.predict(data_val)
        prediction_2d = prediction[0]
        prediction_1d = prediction_2d[0]
        actual_prediction = int(prediction_1d * 1000)
        print(actual_prediction)
    return render_template('/predictions/supportive_forecasting.html', year=year,actual_prediction=actual_prediction, activity_=activity_)

@blueprint.route('/model_samples')
@login_required
def model_samples():
    return render_template('/models/model_samples.html')
