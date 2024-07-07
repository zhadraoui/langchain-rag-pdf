# Projet de Traitement de PDF et de Questions/Réponses

Ce projet utilise `LangChain` et `Streamlit` pour traiter des documents PDF, créer des vecteurs de documents, et permettre une interface interactive de questions/réponses.

##configuration environnement

### creation d'un nouveau environnement personnalisé
conda create -n langchain_rag_env python=3.10

### activation du nouveau environnement
conda activate langchain_rag_env

### installation des dépendances requiquese
pip install -r requirements.txt




## tp01.py

### Améliorations apportées

#### Fonctions séparées

- `load_and_split_pdf(url)`: Charge et divise le PDF en segments.
- `initialize_vectorstore(documents)`: Initialise le vectorstore avec les documents fournis.
- `create_prompt_template()`: Crée le modèle de prompt.
- `create_retrieval_qa_chain(vectorstore, prompt_template)`: Crée la chaîne de QA de récupération.
- `main()`: Fonction principale pour orchestrer le flux global.

#### Gestion des exceptions

Ajout de blocs `try-except` pour gérer les erreurs et fournir des messages d'erreur utiles.

#### Commentaires

Ajout de commentaires explicatifs pour clarifier chaque section du code.

#### Améliorations des performances

Utilisation d'objets globaux uniquement lorsque nécessaire pour réduire les redondances.

### Résultat

En suivant ces améliorations, le code devient plus structuré, maintenable et résilient aux erreurs.

## tp02.py

### Intégration de Streamlit

Pour intégrer Streamlit à votre code existant, utilisez la commande suivante pour installer Streamlit :

```sh
pip install streamlit


Instructions pour exécuter l'application Streamlit :
1. Enregistrez le script ci-dessus dans un fichier, par exemple app.py.
3. Exécutez l'application Streamlit avec la commande suivante :
###streamlit run tp02.py


Explication des modifications :
Intégration de Streamlit :

1. Ajout de l'importation de streamlit (sous le nom st).
2. Utilisation des widgets Streamlit pour entrer l'URL du PDF, les requêtes, et afficher les résultats.
3. Utilisation de st.error(), st.warning(), st.success(), et st.write() pour afficher des messages et des résultats à l'utilisateur.
4. Gestion de l'état :

Utilisation de st.session_state pour stocker le vectorstore entre les interactions utilisateur.
Interface utilisateur :

1. Champs de saisie pour l'URL du PDF et les requêtes.
2. Boutons pour charger le PDF, initialiser le vectorstore, et soumettre les requêtes.
3. Affichage des réponses dans l'interface Streamlit.

A prompt will appear, where questions may be asked:

```
Query: How many locations does WeWork have?
```
