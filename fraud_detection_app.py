import numpy as np
import pickle
from urllib.parse import urlparse
import ipaddress
import re
from bs4 import BeautifulSoup
import urllib
import urllib.request
from datetime import datetime
import requests
import whois
import tldextract
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
import datetime
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the LSTM model
model = load_model('seq2seq_model.h5')

# Load the pre-fitted scaler (assuming it was saved previously)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define feature extraction functions
def havingIP(url):
    try:
        ipaddress.ip_address(url)
        ip = 1
    except:
        ip = 0
    return ip

def haveAtSign(url):
    return 1 if "@" in url else 0

def getLength(url):
    return 1 if len(url) >= 54 else 0

def getDepth(url):
    s = urlparse(url).path.split('/')
    depth = 0
    for j in range(len(s)):
        if len(s[j]) != 0:
            depth += 1
    return depth

def redirection(url):
    pos = url.rfind('//')
    if pos > 6:
        return 1 if pos > 7 else 0
    else:
        return 0

def httpDomain(url):
    domain = urlparse(url).netloc
    return 1 if 'https' in domain else 0

shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|qr\.ae|adf\.ly|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net"

def tinyURL(url):
    return 1 if re.search(shortening_services, url) else 0

def prefixSuffix(url):
    return 1 if '-' in urlparse(url).netloc else 0

def web_traffic(url):
    try:
        rank = BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + url, timeout=10).read(), "xml").find("REACH")['RANK']
        rank = int(rank)
    except:
        return 1
    return 0 if rank < 100000 else 1

def domainAge(domain_name):
    try:
        creation_date = domain_name.creation_date
        expiration_date = domain_name.expiration_date
        if isinstance(creation_date, str) or isinstance(expiration_date, str):
            creation_date = datetime.strptime(creation_date, '%Y-%m-%d')
            expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")
        if expiration_date is None or creation_date is None:
            return 1
        ageofdomain = abs((expiration_date - creation_date).days)
        return 1 if (ageofdomain / 30) < 6 else 0
    except:
        return 1

def domainEnd(domain_name):
    try:
        expiration_date = domain_name.expiration_date
        if isinstance(expiration_date, str):
            expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")
        if expiration_date is None:
            return 1
        today = datetime.now()
        end = abs((expiration_date - today).days)
        return 0 if (end / 30) < 6 else 1
    except:
        return 1

def iframe(response):
    try:
        return 1 if re.findall(r"[<iframe>|<frameBorder>]", response.text) else 0
    except:
        return 1

def mouseOver(response):
    try:
        return 1 if re.findall("<script>.+onmouseover.+</script>", response.text) else 0
    except:
        return 1

def rightClick(response):
    try:
        return 1 if re.findall(r"event.button ?== ?2", response.text) else 0
    except:
        return 1

def forwarding(response):
    try:
        return 0 if len(response.history) <= 2 else 1
    except:
        return 1

def featureExtraction(url):
    features = []

    # Address bar based features
    features.append(havingIP(url))
    features.append(haveAtSign(url))
    features.append(getLength(url))
    features.append(getDepth(url))
    features.append(redirection(url))
    features.append(httpDomain(url))
    features.append(tinyURL(url))
    features.append(prefixSuffix(url))

    # Domain based features
    dns = 0
    try:
        domain_name = whois.whois(urlparse(url).netloc)
    except:
        dns = 1

    features.append(dns)
    features.append(web_traffic(url))
    features.append(1 if dns == 1 else domainAge(domain_name))
    features.append(1 if dns == 1 else domainEnd(domain_name))

    # HTML & Javascript based features
    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.RequestException:
        response = ""

    features.append(iframe(response))
    features.append(mouseOver(response))
    features.append(rightClick(response))
    features.append(forwarding(response))

    return features

# Streamlit app code
def main():
    st.title("Fraud Website Detection")

    # URL input
    url = st.text_input("Enter the website URL:")
    if st.button("Check"):
        if url:
            features = featureExtraction(url)

            # Prepare the features for the model
            feature_names = ['Have_IP', 'Have_At', 'URL_Length', 'URL_Depth', 'Redirection', 
                             'https_Domain', 'TinyURL', 'Prefix_Suffix', 'DNS_Record', 
                             'Web_Traffic', 'Domain_Age', 'Domain_End', 'iFrame', 
                             'Mouse_Over', 'Right_Click', 'Web_Forwards']
            
            df = pd.DataFrame([features], columns=feature_names)

            # Standardize the features
            df_scaled = scaler.transform(df)  # Use transform instead of fit_transform

            # Reshape data for the model [samples, timesteps, features]
            df_reshaped = df_scaled.reshape((df_scaled.shape[0], df_scaled.shape[1], 1))

            # Create initial decoder input sequence
            decoder_input = np.zeros_like(df_reshaped)

            # Make predictions
            predictions = model.predict([df_reshaped, decoder_input])

            # Process the predictions to get a single label
            predicted_label = (predictions.mean(axis=1) > 0.2).astype(int)[0]

            # Display results
            if predicted_label == 1:
                st.error("Warning! This website is Suspicious.")
            else:
                st.success("This website is Safe.")
        else:
            st.warning("Please enter a valid URL.")

if __name__ == "__main__":
    main()
