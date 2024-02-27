import os
import shutil

def merge_folders(folder_1, folder_2, destination_folder_path):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
    
    # Define a function to copy files from source to destination
    def copy_files(source_folder, dest_folder):
        for root, dirs, files in os.walk(source_folder):
            # Construct the destination path
            dest_path = os.path.join(dest_folder, os.path.relpath(root, source_folder))
            
            # Create directories in the destination path
            os.makedirs(dest_path, exist_ok=True)
            
            for file in files:
                file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_path, file)

                # # Check if the file already exists at the destination
                # if os.path.exists(dest_file_path):
                #     # If so, rename the duplicate
                #     basename, extension = os.path.splitext(file)
                #     count = 1
                #     new_basename = f"{basename}_{count}"
                #     new_dest_file_path = os.path.join(dest_path, f"{new_basename}{extension}")
                    
                #     # Increment the counter until we find a new file name
                #     while os.path.exists(new_dest_file_path):
                #         count += 1
                #         new_basename = f"{basename}_{count}"
                #         new_dest_file_path = os.path.join(dest_path, f"{new_basename}{extension}")
                    
                #     # Copy the file with the new name
                #     shutil.copy2(file_path, new_dest_file_path)
                # else:
                    # If not, simply copy the file
                shutil.copy2(file_path, dest_file_path)
    
    # Copy files from both folders to the destination folder
    copy_files(folder_1, destination_folder_path)
    copy_files(folder_2, destination_folder_path)

if __name__ == '__main__':
    merge_folders('data/M2FPA/Test_Bins_Raw/0_30_22.5_45', 'data/M2FPA/Test_Bins_Raw/0_30_-45_-22.5', 'data/M2FPA/Test_Bins_Raw/0_30_+-22.5_+-45')
