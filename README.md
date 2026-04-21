# Popisné modely pre analýzu dát opacientoch s COVID-19

### Šimon Mikolaj
    # ŠKOLITEĽ: prof. Ing. Ján Paralič, PhD.
    # ROK: 2025/2026
    # UNIVERZITA: Technická univerzita v Košiciach
    # FAKULTA: Fakulta Elektrotechniky a Informatiky 
    # ŠTUDIJNÝ ODBOR: Informatika
    # ŠTUDIJNÝ PROGRAM: Hospodárska informatika

Tento repozitár obsahuje zdrojové kódy mojej diplomovej práce zameranej na identifikáciu základných fenotypov pacientov s COVID-19 pomocou algoritmov zhlukovania a na porovnanie ich klinických charakteristík v rámci vĺn pandémie.

## Abstrakt

Zameraním tejto práce je na základe popisných modelov analýzy dát a lekárskych správ pacientov s COVID-19 identifikovať pomocou algoritmov zhlukovania základné fenotypy pacientov a porovnať ich klinické charakteristiky v rámci vĺn pandémie. V práci sú použité a porovnané tri algoritmy, a to k-means, k-medoids a aglomeratívne zhlukovanie. Kvalita zhlukovania je meraná internými validačnými kritériami a to WCSS a Silhouette indexom a tiež čistotou zhlukov. Fenotypy sú porovnávané rôznymi štatistickými testami vrátene post-hoc testov, SHAP metódou a vizualizované PCA metódou. Pre detailnejší pohľad na konkrétnych pacientov, z hadiska ich fenotypov, sú tiež aplikované aj kontrafaktuálne vysvetlenia. Následne bola pre priradenie pacienta do fenotypu a tvorbu kontrafaktuálov, vytvorená a lekármi otestovaná webová aplikácia.

---

##  Obsah repozitára

### Príprava dát a modelovanie (`/data_preparation_and_modeling`)
- Pochopenie a príprava dát
- Modelovanie - algoritmus k-means, k-medoids a aglomeratívne zhlukovanie
- Testovanie optimálneho počtu zhlukov, PCA a SHAP vizualizácia, porovnaie pomocou štatistických testov

###  Webová aplikácia (`/web_application`)
- aplikácia zameraná na priradenie nového pacienta do fenotypu a výpočet jeho kontrafaktuálnych vysvetlení

---

###  Dataset

Dáta o pacientoch s COVID-19 boli poskytnuté v rámci projektu VEGA č. 1/0259/24 z Univerzitnej nemocnice L. Pasteura v Košiciach a v tomto repozitári nie sú zverejnené.

---

###  Použité knižnice

Na spustenie zdrojových kódov sú potrebné nasledujúce knižnice Pythonu:

  - joblib >= 1.5.2 - pre ukladanie modelov
  - lightgbm >= 4.6.0 - pre LightGBM klasifikátor
  - matplotlib >= 3.8.2 - tvorba grafov a vizualizácií
  - miceforest >= 6.0.5 - doplnenie chýbajúcich hodnôt MICE
  - numpy >= 1.26.4 - numerické výpočty a operácie s poľami
  - pandas >= 2.3.3 - manipulácia a predspracovanie dát
  - pyclustering >= 0.10.1.2 - algoritmy zhlukovania a ich metriky
  - re - regulárne výrazy na spracovanie textu (štandardná knižnica)
  - scikit-learn >= 1.7.2 - zhlukovacie algoritmy a metriky
  - scikit_posthocs >= 0.12.0 - post-hoc štatistické testy
  - scipy >= 1.12.0 - dendrogram, rôzne štatistické testy
  - seaborn >= 0.13.2 - tvorba grafov a vizualizácií
  - shap >= 0.45.1 - vysvetľovanie modelov strojového učenia
  - streamlit >= 1.51.0 - vytváranie interaktívnych webových aplikácií