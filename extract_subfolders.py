import os
import time
import logging
import configparser
import pandas as pd
from pathlib import Path
import pybliometrics
import numpy as np
import matplotlib.pyplot as plt
import statistics


try:
    pybliometrics.scopus.init()
    print("✅ Pybliometrics initialized manually!")
except Exception as e:
    print(f"❌ ERROR: Problem with manual initialization of Pybliometrics: {e}")
    exit(1)

from pybliometrics.scopus import AbstractRetrieval

CONFIG_PATH = Path(r"C:\Users\Francesco\.config\pybliometrics.cfg")
if not CONFIG_PATH.exists():
    config = configparser.ConfigParser()
    config["Directories"] = {
        "AbstractRetrieval": r"C:\Users\Francesco\anaconda3\Lib\site-packages\pybliometrics\Scopus\abstract_retrieval",
        "AffiliationRetrieval": r"C:\Users\Francesco\anaconda3\Lib\site-packages\pybliometrics\Scopus\affiliation_retrieval",
        "AffiliationSearch": r"C:\Users\Francesco\anaconda3\Lib\site-packages\pybliometrics\Scopus\affiliation_search",
        "AuthorRetrieval": r"C:\Users\Francesco\anaconda3\Lib\site-packages\pybliometrics\Scopus\author_retrieval",
        "AuthorSearch": r"C:\Users\Francesco\anaconda3\Lib\site-packages\pybliometrics\Scopus\author_search",
        "CitationOverview": r"C:\Users\Francesco\anaconda3\Lib\site-packages\pybliometrics\Scopus\citation_overview",
        "ScopusSearch": r"C:\Users\Francesco\anaconda3\Lib\site-packages\pybliometrics\Scopus\scopus_search",
        "SerialSearch": r"C:\Users\Francesco\anaconda3\Lib\site-packages\pybliometrics\Scopus\serial_search",
        "SerialTitle": r"C:\Users\Francesco\anaconda3\Lib\site-packages\pybliometrics\Scopus\serial_title",
        "PlumXMetrics": r"C:\Users\Francesco\anaconda3\Lib\site-packages\pybliometrics\Scopus\plumx",
        "SubjectClassifications": r"C:\Users\Francesco\anaconda3\Lib\site-packages\pybliometrics\Scopus\subject_classification"
    }
    config["Authentication"] = {
        "APIKey": "ee1866f96d6af76a750def9a35c21d6a,67c95731750e364ef8c277caefa27536,c65f4036ada645d2f2f317e015b41d93,fd46a92d839d6b9ab84aa9cd531d94c2,67a759516292a895f74b9a7f5d9aeecb,82801518de34b2e8e88b96d57c3fc49d,068b755621c6907f2758f275a07fffc9,b2dc4eae2835b79ce18880bba0740580,b5c27402c3a7aea3a5f94596857cc96e,fe2187fcb0f813ddc0b694e09d6eeda2"
    }
    
    
    #Replace the previous APIKeys and the previous pybliometrics paths with the ones used 
    #by the user after having downloading pybliometrics successfully
    
    
    config["Requests"] = {
        "Timeout": "20",
        "Retries": "5"
    }
    with open(CONFIG_PATH, "w") as configfile:
        config.write(configfile)
    print(f"✅ Configuration file successfully created in {CONFIG_PATH}")

os.environ["PYBLIOMETRICS_CONFIG"] = str(CONFIG_PATH)
print(f"✅ Pybliometrics is using the configuration file: {CONFIG_PATH}")

try:
    from pybliometrics.scopus import AbstractRetrieval
    print("✅ Pybliometrics initialized manually!")
except Exception as e:
    print(f"❌ ERROR: Problem with initialization of Pybliometrics: {e}")
    exit(1)
    
mother_folder = r"G:\Drive condivisi\fair index paper SCoSC\science of science\\"  #enter the appropriate mother folder

#folder for all data to include in the analysis
data_folder = mother_folder + "dati\\Dipartimento Fisiologia\\"         #enter the appropriate folder storing your data

#folder for analysis results
results_folder = mother_folder + "risultati\\prova finale\\"              #enter the appropriate folder set to store your results

subfolders = [name for name in os.listdir(data_folder)
                  if os.path.isdir(os.path.join(data_folder, name))]

df_all_authors = pd.DataFrame()
index_mean_all_authors = {
    'Folderpath': [],
    'Author': [], 
    'Index' : [],
    'Std': [], 
    'Papers Number' : []
    }

id = 0

for folder in subfolders:
    id += 1
    
    # ==================== SET THE INPUT FOLDER ====================
    if not subfolders:
        input_path = [data_folder]
    else:
        input_path = data_folder + folder
    
    INPUT_DIR = Path(input_path)
    os.chdir(INPUT_DIR)
    
    # ==================== SET THE OUTPUT FOLDER ====================
    if not subfolders:
        output_path = [results_folder]
    else:
        output_path = results_folder + folder
    
    OUTPUT_DIR = Path(output_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ==================== LOGGING CONFIGURATION ====================
    logging.basicConfig(filename="error_log.txt", level=logging.ERROR,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    # ==================== UNIVERSITIES, REGIONS AND TYPES DICTIONARIES ====================
    # Mapping "base" to search the university names
    UNI_MAPPING = {
        # Group 1: Northern Italy Universities
        "University of Padua": "Università di Padova",
        "University of Verona": "Università di Verona",
        "IUAV University Venice": "Università Iuav di Venezia",
        "Universita Ca Foscari Venezia": "Università Ca' Foscari di Venezia",
        "University of Milan": "Università degli Studi di Milano",
        "University of Milano-Bicocca": "Università degli Studi di Milano-Bicocca",
        "Polytechnic University of Milan": "Politecnico di Milano",
        "University of Bergamo": "Università degli Studi di Bergamo",
        "University of Brescia": "Università degli Studi di Brescia",
        "University of Insubria": "Università degli Studi dell'Insubria",
        "University of Pavia": "Università degli Studi di Pavia",
        "University of Genoa": "Università degli Studi di Genova",
        "University of Bologna": "Università di Bologna",
        "University of Ferrara": "Università degli Studi di Ferrara",
        "University of Parma": "Università degli Studi di Parma",
        "University of Modena e Reggio Emilia": "Università degli Studi di Modena e Reggio Emilia",
        "University of Trieste": "Università degli Studi di Trieste",
        "University of Udine": "Università degli Studi di Udine",
        "University of Turin": "Università degli Studi di Torino",
        "Polytechnic University of Turin": "Politecnico di Torino",
        "University of Trento": "Università degli Studi di Trento",
        "Free University of Bozen-Bolzano": "Università Libera di Bolzano",
        "University of Eastern Piedmont Amedeo Avogadro": "Università degli Studi dell'Amedeo Avogadro",
        "International School for Advanced Studies (SISSA)": "Scuola Internazionale Superiore di Studi Avanzati (SISSA)",
        "IUSS Pavia": "IUSS Pavia",
        "Universita Della Valle D'aosta": "Università della Valle d'Aosta",

        # Group 2: Northern Italy Universities (exceptions)
        "Humanitas University": "Humanitas University",  # non ha traduzione
        "Vita-Salute San Raffaele University": "Università Vita-Salute San Raffaele",
        "Bocconi University": "Università Bocconi",
        "IULM International University Languages & Media": "Università IULM",
        "Catholic University of the Sacred Heart": "Università Cattolica del Sacro Cuore",
        "Universita Ecampus" : "Universita Ecampus",

        # Group 3: Center Universities
        "University of Florence": "Università degli Studi di Firenze",
        "University of Pisa": "Università di Pisa",
        "University of Siena": "Università degli Studi di Siena",
        "University Foreigners Siena": "Università per Stranieri di Siena",
        "Scuola Superiore Sant'Anna": "Scuola Superiore Sant'Anna",
        "Scuola Normale Superiore di Pisa": "Scuola Normale Superiore di Pisa",
        "Marche Polytechnic University": "Politecnico delle Marche",
        "University of Camerino": "Università degli Studi di Camerino",
        "University of Urbino": "Università degli Studi di Urbino",
        "University of Macerata": "Università degli Studi di Macerata",
        "University of Perugia": "Università degli Studi di Perugia",
        "University Foreigners Perugia": "Università per Stranieri di Perugia",
        "Sapienza University Rome": "Sapienza Università di Roma",
        "University of Rome Tor Vergata": "Università di Roma Tor Vergata",
        "Roma Tre University": "Università Roma Tre",
        "Foro Italico University of Rome": "Foro Italico",
        "Tuscia University": "Università degli Studi della Tuscia",
        "University of Cassino": "Università degli Studi di Cassino",

        # Group 3: Center Universities (Rome and its surroundings)
        "University Campus Bio-Medico - Rome Italy": "Università Campus Bio-Medico di Roma",
        "LUISS Guido Carli University": "LUISS Guido Carli",
        "Universita LUMSA": "Università LUMSA",
        "Pegaso Online University": "Pegaso Online University",
        "Guglielmo Marconi University": "Guglielmo Marconi University",
        "Niccolo Cusano Online University" : "Niccolo Cusano Online University",
        "Universita degli Studi di Roma Unitelma Sapienza" : "Universita degli Studi di Roma Unitelma Sapienza",
        "Universita Telematica Mercatorum" : "Universita Telematica Mercatorum",
        "Universita Telematica San Raffaele" : "Universita Telematica San Raffaele",
        "UNINETTUNO":"UNINETTUNO",

        # Group 5: Southern Universities and Islands
        "University of Naples Federico II": "Università degli Studi di Napoli Federico II",
        "Parthenope University Naples": "Università di Napoli Parthenope",
        "University of Naples L'Orientale": "Università degli Studi di Napoli L'Orientale",
        "University of Sannio": "Università degli Studi del Sannio",
        "Universita della Campania Vanvitelli": "Università della Campania Luigi Vanvitelli",
        "University of Basilicata": "Università degli Studi della Basilicata",
        "University of L'Aquila": "Università degli Studi dell'Aquila",
        "University of Teramo": "Università degli Studi di Teramo",
        "G d'Annunzio University of Chieti-Pescara": "Università 'G. d'Annunzio' di Chieti-Pescara",
        "University of Molise": "Università degli Studi del Molise",
        "Universita degli Studi di Bari Aldo Moro": "Università degli Studi di Bari Aldo Moro",
        "University of Foggia": "Università degli Studi di Foggia",
        "Politecnico di Bari": "Politecnico di Bari",
        "University of Salento": "Università del Salento",
        "Universita Mediterranea di Reggio Calabria": "Università Mediterranea di Reggio Calabria",
        "University of Calabria": "Università della Calabria",
        "Magna Graecia University of Catanzaro": "Università Magna Graecia di Catanzaro",
        "University of Palermo": "Università degli Studi di Palermo",
        "University of Catania": "Università degli Studi di Catania",
        "University of Messina": "Università degli Studi di Messina",
        "University of Cagliari": "Università degli Studi di Cagliari",
        "University of Sassari": "Università degli Studi di Sassari",
        "Suor Orsola Benincasa University Naples": "Università Suor Orsola Benincasa di Napoli",
        "Universita Kore di ENNA": "Università Kore di ENNA",
        "Universita Telematica Giustino Fortunato" :"Università Telematica Giustino Fortunato"
    }

    UNI_REGION = {
        # Group 1: Northern italy
        "University of Padua": "Northern Italy",
        "University of Verona": "Northern Italy",
        "IUAV University Venice": "Northern Italy",
        "Universita Ca Foscari Venezia": "Northern Italy",
        "University of Milan": "Northern Italy",
        "University of Milano-Bicocca": "Northern Italy",
        "Polytechnic University of Milan": "Northern Italy",
        "University of Bergamo": "Northern Italy",
        "University of Brescia": "Northern Italy",
        "University of Insubria": "Northern Italy",
        "University of Pavia": "Northern Italy",
        "University of Genoa": "Northern Italy",
        "University of Bologna": "Northern Italy",
        "University of Ferrara": "Northern Italy",
        "University of Parma": "Northern Italy",
        "University of Modena e Reggio Emilia": "Northern Italy",
        "University of Trieste": "Northern Italy",
        "University of Udine": "Northern Italy",
        "University of Turin": "Northern Italy",
        "Polytechnic University of Turin": "Northern Italy",
        "University of Trento": "Northern Italy",
        "Free University of Bozen-Bolzano": "Northern Italy",
        "University of Eastern Piedmont Amedeo Avogadro": "Northern Italy",
        "International School for Advanced Studies (SISSA)": "Northern Italy",
        "IUSS Pavia": "Northern Italy",
        "Universita Della Valle D'aosta": "Northern Italy",
        # Group 2: Some exceptions
        "Humanitas University": "Northern Italy",
        "Vita-Salute San Raffaele University": "Northern Italy",
        "Bocconi University": "Northern Italy",
        "IULM International University Languages & Media": "Northern Italy",
        "Catholic University of the Sacred Heart": "Central Italy",  # come da indicazioni
        # Group 3: Central Italy
        "University of Florence": "Central Italy",
        "University of Pisa": "Central Italy",
        "University of Siena": "Central Italy",
        "University Foreigners Siena": "Central Italy",
        "Scuola Superiore Sant'Anna": "Central Italy",
        "Scuola Normale Superiore di Pisa": "Central Italy",
        "Marche Polytechnic University": "Central Italy",
        "University of Camerino": "Central Italy",
        "University of Urbino": "Central Italy",
        "University of Macerata": "Central Italy",
        "University of Perugia": "Central Italy",
        "University Foreigners Perugia": "Central Italy",
        "Sapienza University Rome": "Central Italy",
        "University of Rome Tor Vergata": "Central Italy",
        "Roma Tre University": "Central Italy",
        "Foro Italico University of Rome": "Central Italy",
        "Tuscia University": "Central Italy",
        "University of Cassino": "Central Italy",
        "University Campus Bio-Medico - Rome Italy": "Central Italy",
        "LUISS Guido Carli University": "Central Italy",
        "Universita LUMSA": "Central Italy",
        "Pegaso Online University": "Central Italy",
        "Guglielmo Marconi University": "Central Italy",
        "Niccolo Cusano Online University": "Central Italy",
        "Universita degli Studi di Roma Unitelma Sapienza" : "Central Italy",
        "Universita Telematica Mercatorum" : "Central Italy",
        "Universita Telematica San Raffaele" : "Central Italy",
        "UNINETTUNO":"Central Italy",

        # Group 4: Southern Italy and Islands
        "University of Naples Federico II": "Southern Italy and Islands",
        "Parthenope University Naples": "Southern Italy and Islands",
        "University of Naples L'Orientale": "Southern Italy and Islands",
        "University of Sannio": "Southern Italy and Islands",
        "Universita della Campania Vanvitelli": "Southern Italy and Islands",
        "University of Basilicata": "Southern Italy and Islands",
        "University of L'Aquila": "Southern Italy and Islands",
        "University of Teramo": "Southern Italy and Islands",
        "G d'Annunzio University of Chieti-Pescara": "Southern Italy and Islands",
        "University of Molise": "Southern Italy and Islands",
        "Universita degli Studi di Bari Aldo Moro": "Southern Italy and Islands",
        "University of Foggia": "Southern Italy and Islands",
        "Politecnico di Bari": "Southern Italy and Islands",
        "University of Salento": "Southern Italy and Islands",
        "Universita Mediterranea di Reggio Calabria": "Southern Italy and Islands",
        "University of Calabria": "Southern Italy and Islands",
        "Magna Graecia University of Catanzaro": "Southern Italy and Islands",
        "University of Palermo": "Southern Italy and Islands",
        "University of Catania": "Southern Italy and Islands",
        "University of Messina": "Southern Italy and Islands",
        "University of Cagliari": "Southern Italy and Islands",
        "University of Sassari": "Southern Italy and Islands",
        "Suor Orsola Benincasa University Naples": "Southern Italy and Islands",
        "Universita Kore di ENNA": "Southern Italy and Islands",
        "Universita Telematica Giustino Fortunato" : "Southern Italy and Islands"
    }
 


    # Dictionary assigning a type to any institution whether it's private or not
    UNI_TYPE = {
        # Northern Italy
        "University of Padua": "public",
        "University of Verona": "public",
        "IUAV University Venice": "public",
        "Universita Ca Foscari Venezia": "public",
        "University of Milan": "public",
        "University of Milano-Bicocca": "public",
        "Polytechnic University of Milan": "public",
        "University of Bergamo": "public",
        "University of Brescia": "public",
        "University of Insubria": "public",
        "University of Pavia": "public",
        "University of Genoa": "public",
        "University of Bologna": "public",
        "University of Ferrara": "public",
        "University of Parma": "public",
        "University of Modena e Reggio Emilia": "public",
        "University of Trieste": "public",
        "University of Udine": "public",
        "University of Turin": "public",
        "Polytechnic University of Turin": "public",
        "University of Trento": "public",
        "Free University of Bozen-Bolzano": "public",
        "University of Eastern Piedmont Amedeo Avogadro": "public",
        "International School for Advanced Studies (SISSA)": "public",
        "IUSS Pavia": "public",
        "Universita Della Valle D'aosta": "private",
        # North - exceptions
        "Humanitas University": "private",
        "Vita-Salute San Raffaele University": "private",
        "Bocconi University": "private",
        "IULM International University Languages & Media": "private",
        "Catholic University of the Sacred Heart": "private",
        # Central Italy
        "University of Florence": "public",
        "University of Pisa": "public",
        "University of Siena": "public",
        "University Foreigners Siena": "public",
        "Scuola Superiore Sant'Anna": "public",
        "Scuola Normale Superiore di Pisa": "public",
        "Marche Polytechnic University": "public",
        "University of Camerino": "public",
        "University of Urbino": "public",
        "University of Macerata": "public",
        "University of Perugia": "public",
        "University Foreigners Perugia": "public",
        "Sapienza University Rome": "public",
        "University of Rome Tor Vergata": "public",
        "Roma Tre University": "public",
        "Foro Italico University of Rome": "public",
        "Tuscia University": "public",
        "University of Cassino": "public",
        "University Campus Bio-Medico - Rome Italy": "private",
        "LUISS Guido Carli University": "private",
        "Universita LUMSA": "private",
        "Pegaso Online University": "private",
        "Guglielmo Marconi University": "private",
        "Niccolo Cusano Online University" :"private",
        "Universita degli Studi di Roma Unitelma Sapienza" : "private",
        "Universita Telematica Mercatorum" : "private",
        "Universita Telematica San Raffaele" : "private",
        "UNINETTUNO":"private",
        # Southern Italy and Islands
        "University of Naples Federico II": "public",
        "Parthenope University Naples": "public",
        "University of Naples L'Orientale": "public",
        "University of Sannio": "public",
        "Universita della Campania Vanvitelli": "public",
        "University of Basilicata": "public",
        "University of L'Aquila": "public",
        "University of Teramo": "public",
        "G d'Annunzio University of Chieti-Pescara": "public",
        "University of Molise": "public",
        "Universita degli Studi di Bari Aldo Moro": "public",
        "University of Foggia": "public", 
        "Politecnico di Bari": "public",
        "University of Salento": "public",
        "Universita Mediterranea di Reggio Calabria": "public",
        "University of Calabria": "public",
        "Magna Graecia University of Catanzaro": "public",
        "University of Palermo": "public",
        "University of Catania": "public",
        "University of Messina": "public",
        "University of Cagliari": "public",
        "University of Sassari": "public",
        "Suor Orsola Benincasa University Naples": "private",
        "Universita Kore di ENNA": "private",
        "Universita Telematica Giustino Fortunato": "private"
    }

    UNI_PRIVATE_TYPE = {  # Dictionary assigning a type to any private institution whether it's telematic or not
        "Universita Della Valle D'aosta": "non-telematic",
        # North-exceptions
        "Humanitas University": "non-telematic",
        "Vita-Salute San Raffaele University": "non-telematic",
        "Bocconi University": "non-telematic",
        "IULM International University Languages & Media": "non-telematic",
        "Catholic University of the Sacred Heart": "non-telematic",
        "University Campus Bio-Medico - Rome Italy": "non-telematic",
        "LUISS Guido Carli University": "non-telematic",
        "Universita LUMSA": "non-telematic",
        "Pegaso Online University": "telematic",
        "Guglielmo Marconi University": "telematic",
        "Niccolo Cusano Online University" : "telematic",
        "Universita degli Studi di Roma Unitelma Sapienza" : "telematic",
        "Universita Telematica Mercatorum" : "telematic",
        "Universita Telematica San Raffaele" : "telematic",
        "Suor Orsola Benincasa University Naples": "non-telematic",
        "Universita Kore di ENNA": "non-telematic",
        "Universita Telematica Giustino Fortunato" :"telematic",
        "UNINETTUNO":"telematic",
        "Universita Ecampus": "telematic",
        "Universita Telematica Giustino Fortunato": "non-telematic"
        }
       
     # ========== IMPORT EXTRA (henceforth only) ==========
    import re
    from difflib import SequenceMatcher
    
    def get_italian_universities_first_last(text: str) -> tuple[str | None, str | None]:
         """
         Returns the Italian university of the FIRST and LAST author (if found) based on UNI_MAPPING. Case-insensitive.
         """
         if not isinstance(text, str):
             return None, None
         parts = [p.strip() for p in text.split(";") if p.strip()]
         if not parts:
             return None, None
    
         first_aff = parts[0]
         last_aff  = parts[-1]
    
         first_univ = next((u for u in UNI_MAPPING if u.lower() in first_aff.lower()), None)
         last_univ  = next((u for u in UNI_MAPPING if u.lower() in last_aff.lower()),  None)
    
         return first_univ, last_univ
    
    

    # ==================== AGGREGATED STRUCTURE PREPARATION ====================
    # To add also "Publication Year", "Source Title" and "University Type" fields
    agg_region_data = {
        "Northern Italy": {
            'Paper Title': [], 'Authors': [], 'Paper DOI': [], 'Cited References': [], 'Number of Authors in Paper': [],
            'Number of Authors in Each Reference': [], 'Index': [], 'Successes': [], 'Failures': [],
            'Italian University First': [], 'Italian University Last': [], 'Autore Italiano': [],
            'Italian University': [], 'Region': [], 'Publication Year': [], 'Source Title': [],
            'University Type': [], 'Private Type': [], 'Reprint Addresses': [], 'Original Doc Type': []
        },
        "Central Italy": {
            'Paper Title': [], 'Authors': [], 'Paper DOI': [], 'Cited References': [], 'Number of Authors in Paper': [],
            'Number of Authors in Each Reference': [], 'Index': [], 'Successes': [], 'Failures': [],
            'Italian University First': [], 'Italian University Last': [], 'Autore Italiano': [],
            'Italian University': [], 'Region': [], 'Publication Year': [], 'Source Title': [],
            'University Type': [], 'Private Type': [], 'Reprint Addresses': [], 'Original Doc Type': []
        },
        "Southern Italy and Islands": {
            'Paper Title': [], 'Authors': [], 'Paper DOI': [], 'Cited References': [], 'Number of Authors in Paper': [],
            'Number of Authors in Each Reference': [], 'Index': [], 'Successes': [], 'Failures': [],
            'Italian University First': [], 'Italian University Last': [], 'Autore Italiano': [],
            'Italian University': [], 'Region': [], 'Publication Year': [], 'Source Title': [],
            'University Type': [], 'Private Type': [], 'Reprint Addresses': [], 'Original Doc Type': []
        }
    }
    
    # ==================== FUNCTION FOR DATA RETRIEVAL USING THE DOI ====================
    def fetch_reference_data(doi, retries=1, wait=0.01):
        """Retrieve an article's data from Scopus using its DOI"""
        for attempt in range(retries):
            try:
                ref = AbstractRetrieval(doi, view="FULL")
                if ref.title is None:
                    raise ValueError("Data not found for this DOI.")
                return ref
            except Exception as e:
                logging.error(f"❌ Attempt {attempt+1} failed for DOI {doi}: {e}")
                print(f"Error in data retrieval for {doi}: {e}")
                time.sleep(wait)
        return None

    # ==================== FILES ELABORATION ====================
    # Input files need to contain the columns: DOI, Authors, Affiliations, Publication Year, Source Title
    excel_files = [f for f in INPUT_DIR.glob("*.xls") if not f.name.startswith("authors_number_per_reference_")]

    if not excel_files:
        #raise SystemExit("❌ ERROR: No Input Excel file found in the folder.")
        print("❌ ERROR: No Input Excel file found in the folder.")
        continue   
    
    # Define the required columns
    required_columns = ["DOI", "Authors", "Affiliations", "Publication Year", "Source Title", "Document Type", "WoS Categories"]

    for file in excel_files:
        print(f"\n🗎 Elaboration file: {file.name}")
        try:
            papers_references = pd.read_excel(file)
        except Exception as e:
            logging.error(f"Error in the opening of the file {file.name}: {e}")
            continue

        if not all(col in papers_references.columns for col in required_columns):
            print(f"ℹ️ File {file.name} does not contain all the required columns; skip.")
            continue
       
        for i, row in papers_references.iterrows():       
            doi = row["DOI"]
            authors_cell = row["Authors"]
            affiliations_cell = row["Affiliations"]
            pub_year = row["Publication Year"]
            source_title = row["Source Title"]
            document_type = row["Document Type"]
            doc_type_raw = document_type 
            wos_category = row["WoS Categories"]
            reprint_addresses = row["Reprint Addresses"]          
            
            try:
                if pub_year < 2001 or pub_year > 2023 :
                    print(f"ℹ️ File {file.name} row {i} : article published out of the considered period; skip.")
                    continue
                    
            except Exception as e:
                    logging.error(f"Publication year absent from data; impossible to classify, skip: {e}")
                    continue
                
            if document_type != 'Article' :
                print(f"ℹ️  File {file.name} row {i}: this document is not a file, skip.")
                continue
                 
            if 'Neurosciences' not in wos_category:
                print(f"ℹ️  File {file.name} row {i}: not a neuroscience article; skip.")
                continue
                            
            # Detects the first University listed in the Affiliations field
            # Only papers in which the last author is italian are kept
            # First/Last author
            first_univ, last_univ = get_italian_universities_first_last(affiliations_cell)
            if not (first_univ or last_univ):
                 continue

            # Concatenation and role
            if first_univ and last_univ:
                italian_univ = f"{first_univ}; {last_univ}"
                role = "Both"
            elif last_univ:
                italian_univ, role = last_univ, "Last"
            else:
                italian_univ, role = first_univ, "First"

            key_univ = last_univ or first_univ
            # ===== CHECK =====
            region = UNI_REGION.get(key_univ)
            if region is None or region not in agg_region_data:
                print(f"ℹ️ File {file.name} row {i}: region not detected for '{key_univ}'; skip.")
                continue
            # ======================
            uni_type = UNI_TYPE.get(key_univ, "NA")
            prv_type = UNI_PRIVATE_TYPE.get(key_univ) if uni_type == "private" else None


            print(f"🔍 {file.name} - row {i}: DOI: {doi} - Univ detected: {key_univ} ({region}, {uni_type}, {prv_type})")
            ref_data = fetch_reference_data(doi)
            if ref_data:
                title = ref_data.title
                scopus_authors = "; ".join([f"{author.given_name} {author.surname}" for author in ref_data.authors]) if ref_data.authors else "N/A"
                references = ref_data.references if ref_data.references else "N/A"

                # Method to evaluate the average number of authors in references:
                if ref_data.references and hasattr(ref_data, "references"):
                    list_len = []
                    for ref in ref_data.references:
                        if hasattr(ref, "authors") and ref.authors:
                            # Suddivide la stringa degli autori in base a ";"
                            authors_list = [a.strip() for a in ref.authors.split(";") if a.strip()]
                            num_authors = len(authors_list)
                            list_len.append(num_authors)
                else:
                    list_len = []
                authors_num_ref_average = sum(list_len) / len(list_len) if list_len else None

                authors_num_paper = len(scopus_authors.split(";"))  if isinstance(scopus_authors, str) else 0
                
                if authors_num_paper > 40 :
                    print(f"ℹ️ File {file.name} row {i}: this document contains more than fourty authors; skip.")
                    continue
                
                index_value = (authors_num_paper - authors_num_ref_average) / (authors_num_paper + authors_num_ref_average) if authors_num_ref_average is not None else None

                # Region Data are accumulated
                agg_region_data[region]['Paper Title'].append(title)
                agg_region_data[region]['Authors'].append(scopus_authors)
                agg_region_data[region]['Paper DOI'].append(doi)
                agg_region_data[region]['Cited References'].append(references)
                agg_region_data[region]['Number of Authors in Paper'].append(authors_num_paper)
                agg_region_data[region]['Number of Authors in Each Reference'].append(authors_num_ref_average)
                agg_region_data[region]['Index'].append(index_value)
                agg_region_data[region]['Successes'].append(1)
                agg_region_data[region]['Failures'].append(0)
                # data concatenation 
                agg_region_data[region]['Italian University'].append(italian_univ)
                agg_region_data[region]['Italian University First'].append(first_univ)
                agg_region_data[region]['Italian University Last'].append(last_univ)
                agg_region_data[region]['Autore Italiano'].append(role)
                agg_region_data[region]['Region'].append(region)
                agg_region_data[region]['Publication Year'].append(pub_year)
                agg_region_data[region]['Source Title'].append(source_title)
                agg_region_data[region]['University Type'].append(uni_type)
                agg_region_data[region]['Private Type'].append(prv_type)
                agg_region_data[region]['Reprint Addresses'].append(reprint_addresses)
                agg_region_data[region]['Original Doc Type'].append(doc_type_raw)
            else:
                # in case of failure, append None per each field 
                agg_region_data[region]['Paper Title'].append(None)
                agg_region_data[region]['Authors'].append(None)
                agg_region_data[region]['Paper DOI'].append(doi)
                agg_region_data[region]['Cited References'].append(None)
                agg_region_data[region]['Number of Authors in Paper'].append(None)
                agg_region_data[region]['Number of Authors in Each Reference'].append(None)
                agg_region_data[region]['Index'].append(None)
                agg_region_data[region]['Successes'].append(0)
                agg_region_data[region]['Failures'].append(1)
                agg_region_data[region]['Italian University'].append(None)
                agg_region_data[region]['Italian University First'].append(None)
                agg_region_data[region]['Italian University Last'].append(None)
                agg_region_data[region]['Autore Italiano'].append(None)
                agg_region_data[region]['Region'].append(None)
                agg_region_data[region]['Publication Year'].append(None)
                agg_region_data[region]['Source Title'].append(None)
                agg_region_data[region]['University Type'].append(None)
                agg_region_data[region]['Private Type'].append(None)
                agg_region_data[region]['Reprint Addresses'].append(None)
                agg_region_data[region]['Original Doc Type'].append(None)
                

    # Combine all data in a single DataFrame
    df_all = pd.concat([pd.DataFrame(agg_region_data[r]) for r in agg_region_data], ignore_index=True)
    
    if df_all.empty:
        print(f"ℹ️ No data for the author {folder}, no file created.")
        continue
    
    elif len(df_all) < 10:
        print(f"ℹ️ For {folder} there are fewer articles meeting the conditions required by the set threshold, skip.")
        continue
    
    index_mean_all_authors['Folderpath'].append(input_path)
    index_mean_all_authors['Author'].append(folder)
    index_mean_all_authors['Papers Number'].append(len(df_all))    
    index_values = df_all["Index"].dropna()
    index_values_list = index_values.tolist()
    if len(index_values_list) > 0 :
       # mean_index = sum(index_values_list) / len(index_values_list)
       mean_index = statistics.mean(index_values_list) 
       std_deviation = statistics.stdev(index_values_list)
    else:
       mean_index = 'nan'
       std_deviation = 'nan'
       
    index_mean_all_authors['Index'].append(mean_index)
    index_mean_all_authors['Std'].append(std_deviation)
    df_all_authors = pd.concat([df_all_authors, df_all], ignore_index=True)

    # Saves a single Excel file including all data (Publication Year and Source Title)
    all_output_path = OUTPUT_DIR / "authors_number_per_reference_all.xlsx"
    df_all.to_excel(all_output_path, index=False)
    print(f"✅ File excel containing all data saved in: {all_output_path}")
    
    # Saves Excel Files for each region
    for reg, data_dict in agg_region_data.items():
        if not data_dict['Paper DOI']:
            print(f"ℹ️ No data for the region {reg}, no file created.")
            continue
        df_reg = pd.DataFrame(data_dict)
        output_path = OUTPUT_DIR / f"authors_number_per_reference_{reg}.xlsx"
        df_reg.to_excel(output_path, index=False)
        print(f"✅ File salvato per la regione {reg} in: {output_path}")

    # ==================== SEPARATION BY UNIVERSITY TYPE (public VS private) ====================
    df_pub = df_all[df_all["University Type"] == "public"]
    df_priv = df_all[df_all["University Type"] == "private"]

    # ==================== SEPARATION BY UNIVERSITY TYPE (NON TELEMATIC VS TELEMATIC) ====================
    df_nontele = df_all[df_all["Private Type"] == "non-telematic"]
    df_tele = df_all[df_all["Private Type"] == "telematic"]

    pub_output_path = OUTPUT_DIR / "authors_number_per_reference_public.xlsx"
    priv_output_path = OUTPUT_DIR / "authors_number_per_reference_private.xlsx"
    nontele_output_path = OUTPUT_DIR / "authors_number_per_reference_nontelematic.xlsx"
    tele_output_path = OUTPUT_DIR / "authors_number_per_reference_telematic.xlsx"
    df_pub.to_excel(pub_output_path, index=False)
    df_priv.to_excel(priv_output_path, index=False)
    df_nontele.to_excel(nontele_output_path, index=False)
    df_tele.to_excel(tele_output_path, index=False)
    print(f"✅ Excel File for Public institutions saved in: {pub_output_path}")
    print(f"✅ Excel File for Private institutions saved in: {priv_output_path}")
    print(f"✅ Excel File for Non-Telematic institutions saved in: {nontele_output_path}")
    print(f"✅ Excel File for Telematic institutions saved in: {tele_output_path}")

    
# Saves a single Excel file including all data (also Publication Year and Source Title)
OUTPUT_DIR = Path(results_folder)
all_output_path = OUTPUT_DIR / "authors_number_per_reference_all_authors.xlsx"
df_all_authors.to_excel(all_output_path, index = False)
print(f"✅ Excel File including all authors' data saved in': {all_output_path}")

index_mean_all_authors = pd.DataFrame(index_mean_all_authors)
all_indeces_output_path = OUTPUT_DIR / "average_index_per_all_author.xlsx"
index_mean_all_authors.to_excel(all_indeces_output_path, index=False)
print(f"✅  Excel File including all authors' average indices saved in: {all_output_path}")
 

