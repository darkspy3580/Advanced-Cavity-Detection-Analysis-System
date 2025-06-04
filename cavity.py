import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import zipfile
import os
import tempfile
import shutil
from PIL import Image
import io

st.set_page_config(page_title="Cavity Detection", layout="wide")

class CavityAnalyzer:
    def __init__(self):
        self.temp_dir = None
    
    def create_temp_directory(self):
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        return self.temp_dir
    
    def cleanup_temp_directory(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def extract_zip_file(self, zip_file):
        """Extract ZIP file and organize MIP/NIP images"""
        temp_dir = self.create_temp_directory()
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())
        
        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        mip_images = []
        nip_images = []
        
        # Search for image files in extracted directory
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    folder_name = os.path.basename(root).lower()
                    file_name_lower = file.lower()
                    
                    # Check if it's MIP or NIP based on folder name or filename
                    if 'mip' in folder_name or 'mip' in file_name_lower:
                        mip_images.append(file_path)
                    elif 'nip' in folder_name or 'nip' in file_name_lower:
                        nip_images.append(file_path)
                    else:
                        # If unclear, ask user or use a default classification
                        if len(mip_images) <= len(nip_images):
                            mip_images.append(file_path)
                        else:
                            nip_images.append(file_path)
        
        return mip_images, nip_images
    
    def load_image_from_path(self, image_path):
        """Load image from file path"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image
    
    def denoise_image(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10)
        return denoised
    
    def detect_contours_and_visualize(self, image, min_area=15):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
        
        # Create visualization
        output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(output_img, filtered_contours, -1, (0, 255, 0), 2)
        
        return filtered_contours, output_img, thresh
    
    def extract_cavity_features(self, contours, filename, label):
        cavity_features = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h != 0 else 0
            extent = area / (w * h) if (w * h) != 0 else 0
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area != 0 else 0
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
            
            cavity_features.append([
                filename, area, perimeter, aspect_ratio, extent,
                solidity, circularity, label
            ])
        
        return cavity_features

def main():
    st.title("🦷 Advanced Cavity Detection & Analysis System")
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = CavityAnalyzer()
    
    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Process Images", "Image Analysis", "Cavity Classification", "Results & Visualization"])
    
    with tab1:
        upload_and_process_images()
    
    with tab2:
        if 'analysis_results' in st.session_state:
            image_analysis()
        else:
            st.warning("Please upload and process images first")
    
    with tab3:
        if 'analysis_results' in st.session_state:
            cavity_classification()
        else:
            st.warning("Please analyze images first")
    
    with tab4:
        if 'classification_results' in st.session_state:
            results_visualization()
        else:
            st.warning("Please complete cavity classification first")

def upload_and_process_images():
    st.header("Upload & Process Images")
    
    upload_method = st.radio("Choose upload method:", ["Individual Images", "ZIP File"])
    
    if upload_method == "ZIP File":
        upload_zip_file()
    else:
        upload_individual_images()

def upload_zip_file():
    st.subheader("ZIP File Upload")
    st.info("Upload a ZIP file containing MIP and NIP images. Images should be in folders named 'MIP' and 'NIP' or have 'mip'/'nip' in their filenames.")
    
    zip_file = st.file_uploader("Upload ZIP file", type=['zip'])
    
    if zip_file:
        analyzer = st.session_state.analyzer
        
        try:
            with st.spinner("Extracting ZIP file..."):
                mip_paths, nip_paths = analyzer.extract_zip_file(zip_file)
            
            if mip_paths or nip_paths:
                st.success(f"✅ Extracted {len(mip_paths)} MIP images and {len(nip_paths)} NIP images")
                
                # Store paths for processing
                st.session_state.image_paths = {
                    'mip': mip_paths,
                    'nip': nip_paths
                }
                
                # Preview extracted images
                if mip_paths and nip_paths:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**MIP Sample:**")
                        mip_image = cv2.imread(mip_paths[0])
                        st.image(cv2.cvtColor(mip_image, cv2.COLOR_BGR2RGB), 
                                caption=os.path.basename(mip_paths[0]), use_container_width=True)
                    
                    with col2:
                        st.write("**NIP Sample:**")
                        nip_image = cv2.imread(nip_paths[0])
                        st.image(cv2.cvtColor(nip_image, cv2.COLOR_BGR2RGB), 
                                caption=os.path.basename(nip_paths[0]), use_container_width=True)
                
                # Process images
                process_images_from_paths()
            else:
                st.error("No valid images found in ZIP file")
                
        except Exception as e:
            st.error(f"Error processing ZIP file: {str(e)}")

def upload_individual_images():
    st.subheader("Individual Image Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("MIP Images")
        mip_files = st.file_uploader("Upload MIP images", 
                                   type=['png', 'jpg', 'jpeg'], 
                                   accept_multiple_files=True, 
                                   key="mip")
    
    with col2:
        st.subheader("NIP Images")  
        nip_files = st.file_uploader("Upload NIP images", 
                                   type=['png', 'jpg', 'jpeg'], 
                                   accept_multiple_files=True, 
                                   key="nip")
    
    if mip_files and nip_files:
        st.session_state.uploaded_images = {
            'mip': mip_files,
            'nip': nip_files
        }
        st.success(f"✅ Uploaded {len(mip_files)} MIP images and {len(nip_files)} NIP images")
        
        # Preview uploaded images
        col1, col2 = st.columns(2)
        with col1:
            st.write("**MIP Sample:**")
            mip_image = Image.open(mip_files[0])
            st.image(mip_image, caption=mip_files[0].name, use_container_width=True)
        
        with col2:
            st.write("**NIP Sample:**")
            nip_image = Image.open(nip_files[0])
            st.image(nip_image, caption=nip_files[0].name, use_container_width=True)
        
        # Process uploaded images
        process_uploaded_images()

def process_images_from_paths():
    st.subheader("Image Processing Configuration")
    
    min_area = st.slider("Minimum cavity area threshold", 5, 50, 15)
    
    if st.button("Process Images from ZIP", type="primary"):
        analyzer = st.session_state.analyzer
        mip_paths = st.session_state.image_paths['mip']
        nip_paths = st.session_state.image_paths['nip']
        
        results = {'mip': {}, 'nip': {}}
        all_features = []
        
        progress_bar = st.progress(0)
        total_files = len(mip_paths) + len(nip_paths)
        current_file = 0
        
        # Process MIP images
        st.subheader("Processing MIP Images")
        for mip_path in mip_paths:
            current_file += 1
            progress_bar.progress(current_file / total_files)
            
            try:
                # Load and process image
                image = analyzer.load_image_from_path(mip_path)
                filename = os.path.basename(mip_path)
                
                # Denoise
                denoised = analyzer.denoise_image(image)
                
                # Detect contours and create visualization
                contours, contour_img, thresh = analyzer.detect_contours_and_visualize(denoised, min_area)
                
                # Extract features
                features = analyzer.extract_cavity_features(contours, filename, "MIP")
                all_features.extend(features)
                
                # Store results
                results['mip'][filename] = {
                    'original': image,
                    'denoised': denoised,
                    'contours': contour_img,
                    'threshold': thresh,
                    'cavity_count': len(contours),
                    'features': features
                }
                
                # Display result
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                            caption=f"Original: {filename}")
                with col2:
                    st.image(denoised, caption="Denoised")
                with col3:
                    st.image(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB), 
                            caption=f"Cavities: {len(contours)}")
            except Exception as e:
                st.error(f"Error processing {mip_path}: {str(e)}")
        
        # Process NIP images
        st.subheader("Processing NIP Images")
        for nip_path in nip_paths:
            current_file += 1
            progress_bar.progress(current_file / total_files)
            
            try:
                # Load and process image
                image = analyzer.load_image_from_path(nip_path)
                filename = os.path.basename(nip_path)
                
                # Denoise
                denoised = analyzer.denoise_image(image)
                
                # Detect contours and create visualization
                contours, contour_img, thresh = analyzer.detect_contours_and_visualize(denoised, min_area)
                
                # Extract features
                features = analyzer.extract_cavity_features(contours, filename, "NIP")
                all_features.extend(features)
                
                # Store results
                results['nip'][filename] = {
                    'original': image,
                    'denoised': denoised,
                    'contours': contour_img,
                    'threshold': thresh,
                    'cavity_count': len(contours),
                    'features': features
                }
                
                # Display result
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                            caption=f"Original: {filename}")
                with col2:
                    st.image(denoised, caption="Denoised")
                with col3:
                    st.image(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB), 
                            caption=f"Cavities: {len(contours)}")
            except Exception as e:
                st.error(f"Error processing {nip_path}: {str(e)}")
        
        # Store results and features
        st.session_state.analysis_results = results
        st.session_state.all_features = all_features
        
        progress_bar.progress(1.0)
        st.success("✅ Processing Complete!")
        
        # Summary
        display_processing_summary(results)

def process_uploaded_images():
    st.subheader("Image Processing Configuration")
    
    min_area = st.slider("Minimum cavity area threshold", 5, 50, 15, key="individual_threshold")
    
    if st.button("Process Uploaded Images", type="primary"):
        analyzer = st.session_state.analyzer
        mip_files = st.session_state.uploaded_images['mip']
        nip_files = st.session_state.uploaded_images['nip']
        
        results = {'mip': {}, 'nip': {}}
        all_features = []
        
        progress_bar = st.progress(0)
        total_files = len(mip_files) + len(nip_files)
        current_file = 0
        
        # Process MIP images
        st.subheader("Processing MIP Images")
        for mip_file in mip_files:
            current_file += 1
            progress_bar.progress(current_file / total_files)
            
            # Load and process image
            image = np.array(Image.open(mip_file))
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Denoise
            denoised = analyzer.denoise_image(image)
            
            # Detect contours and create visualization
            contours, contour_img, thresh = analyzer.detect_contours_and_visualize(denoised, min_area)
            
            # Extract features
            features = analyzer.extract_cavity_features(contours, mip_file.name, "MIP")
            all_features.extend(features)
            
            # Store results
            results['mip'][mip_file.name] = {
                'original': image,
                'denoised': denoised,
                'contours': contour_img,
                'threshold': thresh,
                'cavity_count': len(contours),
                'features': features
            }
            
            # Display result
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                        caption=f"Original: {mip_file.name}")
            with col2:
                st.image(denoised, caption="Denoised")
            with col3:
                st.image(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB), 
                        caption=f"Cavities: {len(contours)}")
        
        # Process NIP images
        st.subheader("Processing NIP Images")
        for nip_file in nip_files:
            current_file += 1
            progress_bar.progress(current_file / total_files)
            
            # Load and process image
            image = np.array(Image.open(nip_file))
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Denoise
            denoised = analyzer.denoise_image(image)
            
            # Detect contours and create visualization
            contours, contour_img, thresh = analyzer.detect_contours_and_visualize(denoised, min_area)
            
            # Extract features
            features = analyzer.extract_cavity_features(contours, nip_file.name, "NIP")
            all_features.extend(features)
            
            # Store results
            results['nip'][nip_file.name] = {
                'original': image,
                'denoised': denoised,
                'contours': contour_img,
                'threshold': thresh,
                'cavity_count': len(contours),
                'features': features
            }
            
            # Display result
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                        caption=f"Original: {nip_file.name}")
            with col2:
                st.image(denoised, caption="Denoised")
            with col3:
                st.image(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB), 
                        caption=f"Cavities: {len(contours)}")
        
        # Store results and features
        st.session_state.analysis_results = results
        st.session_state.all_features = all_features
        
        progress_bar.progress(1.0)
        st.success("✅ Processing Complete!")
        
        # Summary
        display_processing_summary(results)

def display_processing_summary(results):
    total_mip_cavities = sum([r['cavity_count'] for r in results['mip'].values()])
    total_nip_cavities = sum([r['cavity_count'] for r in results['nip'].values()])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MIP Images", len(results['mip']))
    with col2:
        st.metric("NIP Images", len(results['nip']))
    with col3:
        st.metric("Total MIP Cavities", total_mip_cavities)
    with col4:
        st.metric("Total NIP Cavities", total_nip_cavities)

def image_analysis():
    st.header("Image Analysis - Side-by-Side Comparison")
    
    results = st.session_state.analysis_results
    
    # Get all image names
    all_mip_names = list(results['mip'].keys())
    all_nip_names = list(results['nip'].keys())
    
    # Interactive image selection
    col1, col2 = st.columns(2)
    with col1:
        selected_mip = st.selectbox("Select MIP Image:", all_mip_names, key="mip_select")
    with col2:
        selected_nip = st.selectbox("Select NIP Image:", all_nip_names, key="nip_select")
    
    if selected_mip and selected_nip:
        display_detailed_comparison(results, selected_mip, selected_nip)
    
    # Cavity count summary table
    st.subheader("Cavity Count Summary")
    summary_data = []
    
    for name, data in results['mip'].items():
        summary_data.append({"Image": name, "Type": "MIP", "Cavity Count": data['cavity_count']})
    
    for name, data in results['nip'].items():
        summary_data.append({"Image": name, "Type": "NIP", "Cavity Count": data['cavity_count']})
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Overall comparison chart
    st.subheader("Overall Comparison")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Total cavities comparison
    mip_total = sum([r['cavity_count'] for r in results['mip'].values()])
    nip_total = sum([r['cavity_count'] for r in results['nip'].values()])
    
    ax1.bar(['MIP', 'NIP'], [mip_total, nip_total], color=['lightblue', 'lightcoral'])
    ax1.set_title('Total Cavities: MIP vs NIP')
    ax1.set_ylabel('Number of Cavities')
    
    # Individual comparison
    mip_counts = [r['cavity_count'] for r in results['mip'].values()]
    nip_counts = [r['cavity_count'] for r in results['nip'].values()]
    
    x = np.arange(max(len(mip_counts), len(nip_counts)))
    
    if len(mip_counts) < len(x):
        mip_counts.extend([0] * (len(x) - len(mip_counts)))
    if len(nip_counts) < len(x):
        nip_counts.extend([0] * (len(x) - len(nip_counts)))
    
    width = 0.35
    ax2.bar(x - width/2, mip_counts[:len(x)], width, label='MIP', color='lightblue')
    ax2.bar(x + width/2, nip_counts[:len(x)], width, label='NIP', color='lightcoral')
    ax2.set_title('Cavity Count per Image')
    ax2.set_xlabel('Image Index')
    ax2.set_ylabel('Number of Cavities')
    ax2.legend()
    
    st.pyplot(fig)

def display_detailed_comparison(results, mip_file, nip_file):
    mip_data = results['mip'][mip_file]
    nip_data = results['nip'][nip_file]
    
    st.subheader(f"Detailed Comparison: {mip_file} vs {nip_file}")
    
    # Original images
    st.write("**Original Images:**")
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(mip_data['original'], cv2.COLOR_BGR2RGB), 
                caption=f"MIP: {mip_file}", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(nip_data['original'], cv2.COLOR_BGR2RGB), 
                caption=f"NIP: {nip_file}", use_container_width=True)
    
    # Processed images with contours
    st.write("**Cavity Detection Results:**")
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(mip_data['contours'], cv2.COLOR_BGR2RGB), 
                caption=f"MIP Cavities: {mip_data['cavity_count']}", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(nip_data['contours'], cv2.COLOR_BGR2RGB), 
                caption=f"NIP Cavities: {nip_data['cavity_count']}", use_container_width=True)
    
    # Threshold images
    st.write("**Threshold Images:**")
    col1, col2 = st.columns(2)
    with col1:
        st.image(mip_data['threshold'], caption="MIP Threshold", use_container_width=True)
    with col2:
        st.image(nip_data['threshold'], caption="NIP Threshold", use_container_width=True)
    
    # Metrics comparison
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MIP Cavities", mip_data['cavity_count'])
    with col2:
        st.metric("NIP Cavities", nip_data['cavity_count'])
    with col3:
        difference = mip_data['cavity_count'] - nip_data['cavity_count']
        st.metric("Difference", difference, delta=difference)

def cavity_classification():
    st.header("Cavity Classification")
    
    if 'all_features' not in st.session_state or not st.session_state.all_features:
        st.error("No cavity features available for classification")
        return
    
    features = st.session_state.all_features
    
    # Create DataFrame
    columns = ["Image", "Area", "Perimeter", "Aspect_Ratio", "Extent", "Solidity", "Circularity", "Label"]
    df = pd.DataFrame(features, columns=columns)
    
    st.subheader("Feature Overview")
    st.write(f"Total cavities detected: {len(df)}")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Classification settings
    st.subheader("Classification Settings")
    n_clusters = st.slider("Number of cavity types", 2, 5, 3)
    
    if st.button("Perform Classification", type="primary"):
        # Feature extraction and scaling
        feature_cols = ["Area", "Perimeter", "Aspect_Ratio", "Extent", "Solidity", "Circularity"]
        X = df[feature_cols]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cavity_Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Assign cavity types
        if n_clusters == 3:
            cluster_to_type = {0: "Shrunken", 1: "Swollen", 2: "Narrow"}
        elif n_clusters == 2:
            cluster_to_type = {0: "Type A", 1: "Type B"}
        else:
            cluster_to_type = {i: f"Type {i+1}" for i in range(n_clusters)}
        
        df['Cavity_Type'] = df['Cavity_Cluster'].map(cluster_to_type)
        
        # Store classification results
        st.session_state.classification_results = df
        st.session_state.cluster_centers = kmeans.cluster_centers_
        st.session_state.scaler = scaler
        st.session_state.feature_cols = feature_cols
        
        st.success("✅ Classification Complete!")
        
        # Display classification results
        display_classification_results(df)

def display_classification_results(df):
    st.subheader("Classification Results")
    
    # Type counts
    type_counts = df['Cavity_Type'].value_counts()
    
    cols = st.columns(len(type_counts))
    for i, (cavity_type, count) in enumerate(type_counts.items()):
        with cols[i]:
            st.metric(cavity_type, count)
    
    # Classification by image type
    st.subheader("Classification by Image Type")
    crosstab = pd.crosstab(df['Label'], df['Cavity_Type'])
    st.dataframe(crosstab, use_container_width=True)
    
    # Detailed results table
    st.subheader("Detailed Classification Results")
    st.dataframe(df, use_container_width=True)
    
    # Download results
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Classification Results CSV",
        data=csv,
        file_name="cavity_classification_results.csv",
        mime="text/csv"
    )

def results_visualization():
    st.header("Results & Visualization")
    
    if 'classification_results' not in st.session_state:
        st.error("No classification results available")
        return
    
    df = st.session_state.classification_results
    
    # Comprehensive charts and graphs
    st.subheader("Cavity Type Distribution")
    
    # Pie chart for cavity type distribution
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Pie chart
    type_counts = df['Cavity_Type'].value_counts()
    ax1.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Cavity Type Distribution')
    
    # Bar chart for MIP vs NIP comparison
    comparison = df.groupby(['Label', 'Cavity_Type']).size().unstack(fill_value=0)
    comparison.plot(kind='bar', ax=ax2)
    ax2.set_title('Cavity Types: MIP vs NIP Comparison')
    ax2.set_xlabel('Image Type')
    ax2.set_ylabel('Count')
    ax2.legend(title='Cavity Type')
    ax2.tick_params(axis='x', rotation=0)
    
    # Box plot for feature distribution
    feature_cols = st.session_state.feature_cols
    df_melted = df.melt(id_vars=['Cavity_Type'], value_vars=feature_cols, 
                       var_name='Feature', value_name='Value')
    
    # Select a subset of features for better visualization
    key_features = ['Area', 'Circularity', 'Solidity']
    df_key_features = df_melted[df_melted['Feature'].isin(key_features)]
    
    import seaborn as sns
    sns.boxplot(data=df_key_features, x='Feature', y='Value', hue='Cavity_Type', ax=ax3)
    ax3.set_title('Feature Distribution by Cavity Type')
    ax3.tick_params(axis='x', rotation=45)
    
    # Correlation heatmap
    corr_matrix = df[feature_cols].corr()
    im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    ax4.set_xticks(range(len(feature_cols)))
    ax4.set_yticks(range(len(feature_cols)))
    ax4.set_xticklabels(feature_cols, rotation=45, ha='right')
    ax4.set_yticklabels(feature_cols)
    ax4.set_title('Feature Correlation Heatmap')
    
    # Add correlation values to heatmap
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8)
    
    plt.colorbar(im, ax=ax4)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Additional detailed visualizations
    st.subheader("Detailed Feature Analysis")
    
    # Feature distribution by cavity type
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_cols):
        cavity_types = df['Cavity_Type'].unique()
        for cavity_type in cavity_types:
            data = df[df['Cavity_Type'] == cavity_type][feature]
            axes[i].hist(data, alpha=0.7, label=cavity_type, bins=20)
        
        axes[i].set_title(f'{feature} Distribution')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Statistical summaries
    st.subheader("Statistical Summary by Cavity Type")
    
    for cavity_type in df['Cavity_Type'].unique():
        st.write(f"**{cavity_type} Cavities:**")
        subset = df[df['Cavity_Type'] == cavity_type][feature_cols]
        st.dataframe(subset.describe(), use_container_width=True)
    
    # Advanced comparison charts
    st.subheader("Advanced Comparison Analysis")
    
    # Cavity count per image
    cavity_per_image = df.groupby(['Image', 'Label']).size().reset_index(name='Cavity_Count')
    
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot of cavity counts
    mip_data = cavity_per_image[cavity_per_image['Label'] == 'MIP']
    nip_data = cavity_per_image[cavity_per_image['Label'] == 'NIP']
    
    ax1.scatter(range(len(mip_data)), mip_data['Cavity_Count'], 
               color='lightblue', label='MIP', alpha=0.7, s=60)
    ax1.scatter(range(len(nip_data)), nip_data['Cavity_Count'], 
               color='lightcoral', label='NIP', alpha=0.7, s=60)
    ax1.set_title('Cavity Count Distribution Across Images')
    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('Cavity Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Average feature values by cavity type and image type
    avg_features = df.groupby(['Label', 'Cavity_Type'])[feature_cols].mean().reset_index()
    
    # Heatmap of average features
    pivot_data = avg_features.pivot_table(index=['Label', 'Cavity_Type'], 
                                         values=feature_cols)
    
    im2 = ax2.imshow(pivot_data.values, cmap='viridis', aspect='auto')
    ax2.set_xticks(range(len(feature_cols)))
    ax2.set_yticks(range(len(pivot_data.index)))
    ax2.set_xticklabels(feature_cols, rotation=45, ha='right')
    ax2.set_yticklabels([f"{idx[0]}-{idx[1]}" for idx in pivot_data.index])
    ax2.set_title('Average Feature Values by Type')
    
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    st.pyplot(fig3)
    
    # Export comprehensive results
    st.subheader("Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export detailed results
        detailed_csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Download Detailed Results",
            data=detailed_csv,
            file_name="detailed_cavity_analysis.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export summary statistics
        summary_stats = []
        for cavity_type in df['Cavity_Type'].unique():
            subset = df[df['Cavity_Type'] == cavity_type]
            stats = {
                'Cavity_Type': cavity_type,
                'Count': len(subset),
                'MIP_Count': len(subset[subset['Label'] == 'MIP']),
                'NIP_Count': len(subset[subset['Label'] == 'NIP']),
                'Avg_Area': subset['Area'].mean(),
                'Avg_Perimeter': subset['Perimeter'].mean(),
                'Avg_Circularity': subset['Circularity'].mean(),
                'Avg_Solidity': subset['Solidity'].mean()
            }
            summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        summary_csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📈 Download Summary Stats",
            data=summary_csv,
            file_name="cavity_summary_statistics.csv",
            mime="text/csv"
        )
    
    with col3:
        # Export image-wise results
        image_stats = cavity_per_image.to_csv(index=False)
        st.download_button(
            label="🖼️ Download Image Stats",
            data=image_stats,
            file_name="image_wise_cavity_stats.csv",
            mime="text/csv"
        )
    
    # Final summary metrics
    st.subheader("Final Analysis Summary")
    
    total_cavities = len(df)
    mip_cavities = len(df[df['Label'] == 'MIP'])
    nip_cavities = len(df[df['Label'] == 'NIP'])
    unique_types = len(df['Cavity_Type'].unique())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cavities Analyzed", total_cavities)
    with col2:
        st.metric("MIP Cavities", mip_cavities)
    with col3:
        st.metric("NIP Cavities", nip_cavities)
    with col4:
        st.metric("Cavity Types Identified", unique_types)
    
    # Performance insights
    st.subheader("Key Insights")
    
    insights = []
    
    # Most common cavity type
    most_common_type = df['Cavity_Type'].value_counts().index[0]
    most_common_count = df['Cavity_Type'].value_counts().iloc[0]
    insights.append(f"• Most common cavity type: **{most_common_type}** ({most_common_count} cavities)")
    
    # MIP vs NIP comparison
    if mip_cavities > nip_cavities:
        insights.append(f"• MIP images show **{mip_cavities - nip_cavities} more cavities** than NIP images")
    elif nip_cavities > mip_cavities:
        insights.append(f"• NIP images show **{nip_cavities - mip_cavities} more cavities** than MIP images")
    else:
        insights.append("• MIP and NIP images show **equal numbers** of cavities")
    
    # Feature insights
    avg_area = df['Area'].mean()
    avg_circularity = df['Circularity'].mean()
    insights.append(f"• Average cavity area: **{avg_area:.2f} pixels**")
    insights.append(f"• Average circularity: **{avg_circularity:.3f}** (1.0 = perfect circle)")
    
    # Type distribution insight
    type_distribution = df['Cavity_Type'].value_counts(normalize=True) * 100
    for cavity_type, percentage in type_distribution.items():
        insights.append(f"• {cavity_type} cavities: **{percentage:.1f}%** of total")
    
    for insight in insights:
        st.write(insight)

if __name__ == "__main__":
    main()