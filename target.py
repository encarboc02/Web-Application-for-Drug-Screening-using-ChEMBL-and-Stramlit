import streamlit as st
from streamlit_option_menu import option_menu
from chembl_webresource_client.new_client import new_client
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
import random
import base64
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import requests


st.set_page_config(layout="wide")

with st.sidebar:
    selected = st.selectbox(
        "Funciones",
        ["Información del Target", "Reposicionamiento"],
        index=0,
    )


if selected == "Información del Target":
    st.title("Información del Target en ChEMBL")

    # Obtener el ID del target
    proteina_diana_id = st.text_input("Introduce el ID del Target:")

    @st.cache_data
    def obtener_informacion_target(proteina_diana_id):
        try:
            protein_info = new_client.target.get(proteina_diana_id)
            if protein_info:
                # Obtención de compuestos asociados
                compounds = new_client.activity.filter(target_chembl_id=proteina_diana_id).only(
                    "molecule_chembl_id", "molecule_pref_name", "standard_value", "standard_type", "standard_relation",
                    "standard_units", "canonical_smiles"
                ).filter(standard_value__isnull=False)

                st.subheader("Datos del Target:")
                st.write("- Organismo:", protein_info.get('organism', 'None'))
                st.write("- Nombre Preferido:", protein_info.get('pref_name', 'None'))
                st.write("- Flag de Grupo de Especies:", protein_info.get('species_group_flag', 'None'))
                st.write("- Tipo de Target:", protein_info.get('target_type', 'None'))
                st.write("- Tax ID:", protein_info.get('tax_id', 'None'))

                cross_references = protein_info.get('cross_references', [])
                if cross_references:
                    st.markdown("<h4 style='font-size:16px;'>Cross-references:</h4>", unsafe_allow_html=True)
                    for reference in cross_references:
                        st.write("- Tipo:", reference.get('xref_name', 'None'))
                        st.write("- ID:", reference.get('xref_id', 'None'))

                comp_ids = []
                compounds_list = []
                for compound in compounds:
                    if compound["standard_type"] == "IC50":
                        compound_dict = {
                            "Compound ID": compound["molecule_chembl_id"],
                            "Compound Name": compound["molecule_pref_name"],
                            "Standard Value": float(compound["standard_value"]), 
                            "Standard Type": compound["standard_type"],
                            "Standard Relation": compound["standard_relation"],
                            "Standard Units": compound["standard_units"],
                            "Canonical SMILES": compound["canonical_smiles"]
                        }
                        compounds_list.append(compound_dict)
                        comp_ids.append(compound["molecule_chembl_id"])

                df_compounds = pd.DataFrame(compounds_list)
                df_compounds_sorted = df_compounds.sort_values('Standard Value', ascending=True).drop_duplicates(subset='Compound ID')

                def clasificar_bioactividad(ic50):
                    if ic50 <= 1000:
                        return 'Activo'
                    elif ic50 >= 10000:
                        return 'Inactivo'
                    else:
                        return 'Intermedio'

                df_compounds_sorted['Bioactivity Class'] = df_compounds_sorted['Standard Value'].apply(clasificar_bioactividad)
                column_order = ['Compound ID', 'Compound Name', 'Standard Value', 'Standard Type','Standard Relation', 'Standard Units', 'Bioactivity Class', 'Canonical SMILES']
                df_compounds_sorted = df_compounds_sorted[column_order]

                if not df_compounds_sorted.empty:
                    st.subheader("Compuestos asociados:")
                    st.write(df_compounds_sorted)

                    csv_data = df_compounds_sorted.to_csv(index=False, quoting=csv.QUOTE_NONNUMERIC, sep=',')
                    b64 = base64.b64encode(csv_data.encode()).decode() 
                    href = f'<a href="data:file/csv;base64,{b64}" download="Compuestos_asociados.csv">Descargar CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    st.markdown("<h4 style='font-size:16px;'>Matriz de bioactividad:</h4>", unsafe_allow_html=True)
                    df_pivot = df_compounds_sorted.pivot(index='Compound ID', columns='Standard Type', values='Standard Value')
                    if 'IC50' in df_pivot.columns: 
                        df_pivot_sorted = df_pivot.sort_values(by='IC50', ascending=True)

                        plt.figure(figsize=(10, 8))
                        heatmap = sns.heatmap(df_pivot_sorted, cmap="viridis", annot=True, fmt=".4f", linewidths=.5)
                        heatmap.set_title('Matriz de bioactividad compuestos-target (IC50)', fontsize=16)
                        heatmap.set_ylabel('Compuesto ID', fontsize=12)

                        st.pyplot(plt)

                    return df_compounds_sorted, comp_ids

            else:
                st.write("No se encontró información para la proteína diana con el ID proporcionado.")
                return None, None

        except Exception as e:
            st.error(f"Se produjo un error al obtener la información: {e}")
            return None, None


    def obtener_informacion_compuesto(compound_id):
        try:
            compound_info = new_client.molecule.filter(molecule_chembl_id=compound_id)
            if compound_info:
                num_donantes, num_aceptores = calcular_descriptores_enlaces_hidrogeno(compound_info[0]["molecule_structures"]["canonical_smiles"])
                compound_info_table = {
                    "Molecule Name": compound_info[0]["pref_name"],
                    "Molecular Formula": formula_molecular(compound_info[0]["molecule_structures"]["canonical_smiles"]),
                    "Canonical Smiles": compound_info[0]["molecule_structures"]["canonical_smiles"],
                    "Molecule Type": compound_info[0]["molecule_type"],
                    "Black Box Flag": compound_info[0].get("black_box_warning"),
                    "Max Phase": compound_info[0].get("max_phase"),
                    "AlogP": calcular_alogp(compound_info[0]["molecule_structures"]["canonical_smiles"]),
                    "Número de donantes de enlaces de hidrógeno": num_donantes,
                    "Número de aceptores de enlaces de hidrógeno": num_aceptores,        
                }

                compound_smiles = compound_info_table["Canonical Smiles"]
                pharmacophore = identificar_farmacoforo(compound_smiles)
                smiles_derivados = generar_smiles_distintos(compound_smiles, pharmacophore)
                compound_info_table["SMILES Derivados"] = smiles_derivados

                return compound_info_table
            else:
                return None
        except Exception as e:
            st.error(f"Error al obtener información del compuesto: {e}")
            return None

    def calcular_descriptores_enlaces_hidrogeno(canonical_smiles):
        try:
            mol = Chem.MolFromSmiles(canonical_smiles)
            if mol is not None:
                num_donantes = rdMolDescriptors.CalcNumHBD(mol)  
                num_aceptores = rdMolDescriptors.CalcNumHBA(mol)  
                return num_donantes, num_aceptores
            else:
                return None, None
        except Exception as e:
            return None, None

    def formula_molecular(canonical_smiles):
        try:
            mol = Chem.MolFromSmiles(canonical_smiles)
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
            return formula
        except Exception as e:
            return str(e)

    def calcular_alogp(canonical_smiles):
        try:
            mol = Chem.MolFromSmiles(canonical_smiles)
            if mol is not None:
                alogp = Descriptors.MolLogP(mol)
                return alogp
            else:
                return "Imposible crear una molécula a partir del SMILE proporcionado."
        except Exception as e:
            return f"No se pudo calcular AlogP: {str(e)}"

    def identificar_farmacoforo(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                functional_groups = []

                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() == 7:
                        neighbors = [mol.GetAtomWithIdx(n.GetIdx()).GetSymbol() for n in atom.GetNeighbors()]
                        if 'H' in neighbors:
                            functional_groups.append("Amina")

                    if atom.GetAtomicNum() == 6:
                        neighbors = [mol.GetAtomWithIdx(n.GetIdx()).GetSymbol() for n in atom.GetNeighbors()]
                        if 'O' in neighbors and len(neighbors) == 2:
                            functional_groups.append("Carboxilo")

                        if 'C' in neighbors and len(neighbors) == 5:
                            functional_groups.append("Fenilo")

                    if atom.GetAtomicNum() == 8:
                        neighbors = [mol.GetAtomWithIdx(n.GetIdx()).GetSymbol() for n in atom.GetNeighbors()]
                        if 'H' in neighbors:
                            functional_groups.append("Hidroxilo")

                    if atom.GetAtomicNum() == 6:
                        neighbors = [mol.GetAtomWithIdx(n.GetIdx()).GetSymbol() for n in atom.GetNeighbors()]
                        if 'N' in neighbors and 'O' in neighbors:
                            functional_groups.append("Amida")

                return ", ".join(functional_groups) if functional_groups else "No se encontraron grupos funcionales relevantes."
            else:
                return "No se pudo crear una molécula a partir del SMILES proporcionado."
        except Exception as e:
            return f"No se pudo identificar el farmacóforo: {str(e)}"

    def generar_smiles_distintos(compound_smiles, pharmacophore):
        smiles_list = []
        max_attempts = 100  
        while len(smiles_list) < 10 and max_attempts > 0:
            new_smiles = modificar_smiles(compound_smiles, pharmacophore)
            if smile_quimicamente_posible(new_smiles) and new_smiles not in smiles_list:
                smiles_list.append(new_smiles)
            max_attempts -= 1
        return smiles_list

    def modificar_smiles(canonical_smiles, pharmacophore):
        fragments = canonical_smiles.split('.')
        fragment_to_modify = random.choice(fragments)

        if random.random() < 0.5:
            new_fragment = fragment_to_modify + random.choice(["C", "N", "O", "S"])
        else:
            if len(fragment_to_modify) > 1:
                new_fragment = fragment_to_modify[:-1]
            else:
                new_fragment = fragment_to_modify

        new_smiles = '.'.join([new_fragment if frag == fragment_to_modify else frag for frag in fragments])

        return new_smiles

    def smile_quimicamente_posible(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    if proteina_diana_id:
        df_compounds_sorted, comp_ids = obtener_informacion_target(proteina_diana_id)
        if df_compounds_sorted is not None:
            compound_id_seleccionado = st.selectbox("Selecciona un Compound ID:", comp_ids)

            if compound_id_seleccionado:
                compound_info = obtener_informacion_compuesto(compound_id_seleccionado)
                if compound_info:
                    st.subheader("Datos del compuesto seleccionado:")
                    st.write("- Nombre de la molécula:", compound_info["Molecule Name"])
                    st.write("- Fórmula molecular:", compound_info["Molecular Formula"])
                    st.write("- Tipo de molécula:", compound_info["Molecule Type"])
                    st.write("- Black Box Flag:", compound_info["Black Box Flag"])
                    st.write("- Fase máxima:", compound_info["Max Phase"])
                    st.write("- LogP:", compound_info["AlogP"])
                    st.write("- Número de donantes de enlaces de hidrógeno:", compound_info["Número de donantes de enlaces de hidrógeno"])
                    st.write("- Número de aceptores de enlaces de hidrógeno:", compound_info["Número de aceptores de enlaces de hidrógeno"])
                    st.write("- Canonical SMILE:", compound_info["Canonical Smiles"])
                    
                    st.subheader("Estructura química del compuesto asociado seleccionado:")
                    mol = Chem.MolFromSmiles(compound_info["Canonical Smiles"])
                    if mol is not None:
                        img = Draw.MolToImage(mol)
                        ic50_value = float(df_compounds_sorted.loc[df_compounds_sorted['Compound ID'] == compound_id_seleccionado, 'Standard Value'].iloc[0])
                        caption = f"Estructura química original. SMILES: {compound_info['Canonical Smiles']}. IC50: {ic50_value}"
                        st.image(img, caption=caption)

                    st.subheader("SMILES derivados:")
                    df_smiles = pd.DataFrame({"SMILES Derivados": compound_info["SMILES Derivados"]})
                    st.write(df_smiles)

                    csv = df_smiles.to_csv(index=False, quoting=csv.QUOTE_NONNUMERIC, sep=',')
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="Smiles_derivados.csv">Descargar CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    st.subheader("Representación del SMILE elegido")
                    smiles_personalizada = st.text_input("Introduce el SMILE:")
                    if smiles_personalizada:
                        mol_personalizada = Chem.MolFromSmiles(smiles_personalizada)
                        if mol_personalizada is not None:
                            img_personalizada = Draw.MolToImage(mol_personalizada)
                            st.image(img_personalizada, caption="Estructura seleccionada")
                            alogp_personalizada = calcular_alogp(smiles_personalizada)
                            num_donantes, num_aceptores = calcular_descriptores_enlaces_hidrogeno(smiles_personalizada)
                            st.write("- LogP:", alogp_personalizada)
                            st.write("- Número de donadores de enlaces de hidrógeno:", num_donantes)
                            st.write("- Número de aceptores de enlaces de hidrógeno:", num_aceptores)
                        else:
                            st.warning("La Canonical SMILES introducida no es válida.")

elif selected == "Reposicionamiento":
    st.title("Reposicionamiento para un Compuesto")
    compound_id = st.text_input("Introduce el ID del Compuesto:")

    try:
        compound_info = new_client.molecule.filter(molecule_chembl_id=compound_id)

        if compound_info:
            st.subheader("Información del Compuesto:")
            st.write("- Nombre del compuesto:", compound_info[0]["pref_name"])
            st.write("- Tipo de compuesto:", compound_info[0]["molecule_type"])
            st.write("- Fase máxima:", compound_info[0].get("max_phase"))
            st.write("- Canonical SMILE:", compound_info[0]["molecule_structures"]["canonical_smiles"])

            st.subheader("Targets que se ha predicho que interactúan con el compuesto:")
            smiles_input = st.text_input("Introduce un SMILE:")

            if smiles_input:
                try:
                    response = requests.post("https://www.ebi.ac.uk/chembl/target-predictions", json={"smiles": smiles_input})

                    if response.status_code == 200:
                        predictions = response.json()
                        if predictions:
                            st.subheader("Predicciones de targets")

                            df_predictions = pd.DataFrame(predictions)
                            st.write(df_predictions)
                            csv = df_predictions.to_csv(index=False, quoting=csv.QUOTE_NONNUMERIC, sep=',')
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="Targets_predicción.csv">Descargar CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)

                        else:
                            st.write("No se encontraron predicciones para el SMILES proporcionado.")

                    else:
                        st.write("Se produjo un error al realizar la predicción de objetivos biológicos.")

                except Exception as e:
                    st.error(f"Se produjo un error al realizar la predicción de objetivos biológicos: {e}")

        else:
            st.write("No se encontró información para el compuesto con el ID proporcionado.")

    except Exception as e:
        st.error(f"Se produjo un error al obtener la información del compuesto: {e}")

