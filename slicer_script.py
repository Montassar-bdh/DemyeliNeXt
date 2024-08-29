import os
import re


def to_nifti(nodename,
             anonymized_path=r"/home/radio-monastir/Montasser/DemyeliNeXt/Data/Monastir/NO_MS/anonymized/part2"):
    # Get the loaded volume node (replace 'YourVolumeNodeName' with the actual node name)
    volume_node = slicer.util.getNode(nodename)

    dicom_tag_series_description = "0008,103e"
    series_description = slicer.dicomDatabase.fileValue(volume_node.GetStorageNode().GetFileName(),
                                                        dicom_tag_series_description)
    print(series_description)
    if "FLAIR" in series_description.upper():

        # Access the DICOM tags
        dicom_tag_patient_name = "0010,0010"

        dicom_tag_manufacturer = "0008,0070"
        dicom_tag_model = "0008,1090"
        dicom_tag_station_name = "0008,1010"
        dicom_tag_magnetic_field_strength = "0018,0087"
        # Access the DICOM tag for study date
        dicom_tag_study_date = "0008,0020"

        # Get patient name and series description from the volume's DICOM metadata
        patient_name = slicer.dicomDatabase.fileValue(volume_node.GetStorageNode().GetFileName(),
                                                      dicom_tag_patient_name)

        # Get the study date from the volume's DICOM metadata
        study_date = slicer.dicomDatabase.fileValue(volume_node.GetStorageNode().GetFileName(), dicom_tag_study_date)

        # Get additional metadata
        manufacturer = slicer.dicomDatabase.fileValue(volume_node.GetStorageNode().GetFileName(),
                                                      dicom_tag_manufacturer)
        model = slicer.dicomDatabase.fileValue(volume_node.GetStorageNode().GetFileName(), dicom_tag_model)
        station_name = slicer.dicomDatabase.fileValue(volume_node.GetStorageNode().GetFileName(),
                                                      dicom_tag_station_name)
        magnetic_field_strength = slicer.dicomDatabase.fileValue(volume_node.GetStorageNode().GetFileName(),
                                                                 dicom_tag_magnetic_field_strength)

        # Combine patient name and series description
        combined_name = f"{patient_name}_{series_description}_{study_date}"

        # Clean the combined name to be a valid filename (remove/replace invalid characters)
        combined_name_clean = re.sub(r'[^a-zA-Z0-9]', '_', combined_name).upper()

        # Combine additional metadata to create the output folder path
        output_folder_name = f"{manufacturer}_{model}_{station_name}_{magnetic_field_strength}T"

        # Clean the output folder name to be a valid folder name (remove/replace invalid characters)
        output_folder_name_clean = re.sub(r'[^a-zA-Z0-9]', '_', output_folder_name)

        # Define the output folder and filename
        output_folder = os.path.join(anonymized_path, output_folder_name_clean)
        output_filename = f"{combined_name_clean}.nii.gz"

        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Define the full output path
        output_path = os.path.join(output_folder, output_filename)

        # Save the volume as a NIfTI file
        slicer.util.saveNode(volume_node, output_path)

        print(f"Volume saved as: {output_path}")


# Get a list of all volume nodes in the scene
volume_nodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')

# Extract the names of the volume nodes
for volume_node in volume_nodes:
    volume_name = volume_node.GetName()
    # Print the list of volume names
    print("convert volume", volume_name)
    to_nifti(volume_name)
