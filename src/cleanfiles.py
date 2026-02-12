import os

def cleanFiles(directory='.', formats=[], confirm=True):
    """Function to clean a directory by deleting all files except those with specified formats.
    By default, it keeps a set of important file formats.
    Parameters:
    - directory (str): The target directory to clean. Default is the current directory.
    - formats (list): A list of file extensions to keep (e.g., ['.txt', '.log']).
    - confirm (bool): If True, prompts the user for confirmation before deleting files.
    """
    # Keeps a default set of important file formats and the user-specified ones
    default_formats = ['.py', '.ipynb', '.csv', '.fdf', '.out',
                       '.DM', '.FA ', '.XV', '.HSX', '.bands', '.DOS', '.xyz', '.yaml', '.traj']
    joined_formats = default_formats + formats
    # Identify all files in the directory and the files to keep/delete
    allfiles = {item for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item))}
    keepfiles = {file for file in os.listdir(directory) if file.endswith(tuple(joined_formats))}
    delfiles = allfiles - keepfiles
    # Delete the unwanted files after user confirmation
    if not delfiles:
        # If there are no files to delete
        print("No files to delete.")
    else:
        if confirm:
            # Make a warning requiring user confirmation
            print(f"All files in {directory} except the following will be deleted:")
            print('\n'.join(keepfiles))
            confirmation = input("Do you want to proceed with deleting other files? (y/n): ")
            if confirmation.lower() != 'y':
                print("Aborting file deletion.")
                return
        for item in delfiles:
            os.remove(os.path.join(directory, item))
        if confirm:
            print("Files successfully deleted.")