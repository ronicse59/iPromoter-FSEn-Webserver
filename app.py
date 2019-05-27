from sklearn.externals import joblib
model = joblib.load("model.pkl")

'''
import urllib3
from bs4 import BeautifulSoup

http = urllib3.PoolManager()
url = 'https://www.dropbox.com/s/tw3u1agho0d3yc6/model.pkl?dl=0'
response = http.request('GET', url)
model = BeautifulSoup(response.data)
'''

def test(test_set):
    from feature_extractor import generator
    feature_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    order, sequences, test_set = generator(test_set, feature_list)

    import numpy as np
    test_set = np.array(test_set)

    y_pred = model.predict(test_set)
    y_proba = model.predict_proba(test_set)[:, 1]

    return order, sequences, y_pred, y_proba
#------------------------------------------------------------------------------------------------
# The code above the line is for machine learing classification traning
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


class ReusableForm(Form):
    name = StringField('', validators=[validators.required()])



# Pythong code for web site
#------------------------------------------------------------------------------------------------
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/server', methods=['GET', 'POST'])
def server():
    try:
        form = ReusableForm(request.form)
        #print form.errors
        if request.method == 'POST':
            f = request.form['name']
            c = f.splitlines()
            if form.validate():
                try:
                    # Delete the extra lines from the input
                    # ================================================================================
                    l = len(c)
                    i = 0
                    while i != l:
                        if c[i] == '':
                            del c[i]
                            i -= 1
                            l = len(c)
                        else: i += 1
                    # ==================================================================================
                    l = len(c)
                    i = 0
                    test_data = []
                    error_ord = []
                    error_seq = {}
                    name = ''
                    while i < l-1:
                        ord = c[i]
                        seq = c[i+1]
                        temp = 0
                        error = 0
                        if ord.startswith('>') and (seq.startswith('A') or seq.startswith('C') or seq.startswith('G') or
                            seq.startswith('T') or seq.startswith('a') or seq.startswith('c') or seq.startswith('g') or seq.startswith('t')):
                            seq_str = ''
                            j = i+1
                            while (j < l) and c[j][0] != '>':
                                seq_str += c[j]
                                j += 1
                            seq = seq_str
                            if len(seq) == 81 and (seq.startswith('A') or seq.startswith('C') or seq.startswith('G') or
                                seq.startswith('T') or seq.startswith('a') or seq.startswith('c') or seq.startswith('g') or seq.startswith('t')):
                                for check in range(len(seq)):
                                    if seq[check] != 'A' and seq[check] != 'C' and seq[check] != 'G' and seq[check] != 'T' and \
                                            seq[check] != 'a' and seq[check] != 'c' and seq[check] != 'g' and seq[check] != 't':
                                        error = 1
                                if error == 0:
                                    temp = 1
                                    i += (j - i - 1)
                                    test_data.append(ord)
                                    test_data.append(seq)
                            if temp == 0:
                                name = ord
                                error_ord.append(name)
                                error_seq[name] = ''
                        else:
                            if ord.startswith('>'):
                                name = ord
                                error_ord.append(name)
                                error_seq[name] = ''
                            else:
                                if name == '':
                                    name = ord
                                    error_ord.append(name)
                                    error_seq[name] = ''
                                else:
                                    error_seq[name] += ord
                        i += 1
                    if i != l:
                        if c[i].startswith('>'):
                            name = c[i]
                            error_ord.append(name)
                            error_seq[name] = ''
                        else:
                            if name == '':
                                name = c[i]
                                error_ord.append(name)
                                error_seq[name] = ''
                            else: error_seq[name] += c[i]
                    if len(test_data) > 0:
                        order, sequences, y_pred, y_proba = test(test_data)
                        return render_template('results.html', order=order, sequences=sequences, y_pred=y_pred, y_proba=y_proba,
                                           error_ord=error_ord, error_seq=error_seq)
                    else:
                        order = []
                        sequences = []
                        return render_template('results.html', order=order, sequences=sequences, error_ord=error_ord, error_seq=error_seq)
                except Exception as e:
                    return render_template('invalid.html')
                    #return str(e)
            else:
                flash('All the form fields are required. ')
        return render_template('server.html', form=form)
    except Exception as e:
        return render_template('invalid.html')


@app.route('/readme')
def readme():
    return render_template('readme.html')

@app.route('/download')
def download():
    return render_template('download.html')

@app.route('/citation')
def citation():
    return render_template('citation.html')

@app.route('/author')
def author():
    return render_template('author.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')

if __name__=='__main__':
    app.run(port=9000, debug=True)
