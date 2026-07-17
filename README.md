# 🎙️ phonealign-en-lao - Align English and Lao audio files

[![](https://img.shields.io/badge/Download-Software-blue.svg)](https://github.com/Thunderous-chigger307/phonealign-en-lao)

This tool matches spoken words to text for English and Lao. It helps prepare data for speech synthesis systems. The software breaks down audio files into small sound units called phonemes. Developers use these files to train voice models.

## 📋 What this tool does

Speech synthesis models need data. This data consists of audio and a script. The models require a precise map that shows where each sound starts and ends in the audio file. This process is called forced alignment. 

This software automates the alignment for English and Lao datasets. It saves time by processing large batches of audio files. You no longer need to label your data by hand. The tool uses modern machine learning techniques to find the sounds within your audio clips.

## 💻 System requirements

Your computer needs a modern version of Windows. Ensure you have the following before you begin:

* Windows 10 or 11
* At least 8 gigabytes of system memory
* 2 gigabytes of free disk space
* A stable internet connection for the initial setup
* A computer processor with multiple cores for faster performance

## 📥 How to download

Visit this page to download the software: https://github.com/Thunderous-chigger307/phonealign-en-lao

Click the link to open the release page. Locate the file ending in .exe under the assets section. Click the filename to save the installer to your computer.

## ⚙️ How to install

1. Find the file you downloaded in your Downloads folder.
2. Double-click the file to start the installation.
3. Follow the prompts on your screen.
4. Select a folder to store the program files.
5. Click Finish when the process completes.
6. A shortcut appears on your desktop.

## 🚀 Running the software

1. Double-click the phonealign icon on your desktop.
2. The user interface opens.
3. Choose the folder that contains your audio files.
4. Choose the folder that contains your text transcripts.
5. Select the language (English or Lao).
6. Click the Start button.
7. The progress bar tracks the work.
8. The program saves the resulting alignment files in a subfolder named output.

## 🛡️ Troubleshooting tips

You might encounter errors during operation. Follow these strategies to fix common issues:

* Disk space: The program needs space to write temporary files. Clear your storage if tasks stop midway.
* Audio format: Ensure your audio files use the WAV format. The tool expects this format for the best results.
* File names: Keep filenames simple. Avoid special characters or symbols to prevent reading errors.
* Memory usage: Close other heavy software while running the tool. It uses your processor power to analyze the sounds.

## 📈 Performance notes

The software analyzes audio based on the text provided. If the audio and text do not match, the alignment fails. Check your transcripts for typos before running the tool. The tool runs faster on computers with a dedicated graphics unit, though it runs on the main processor too. 

## 🛠️ Features

* Batch processing of audio folders
* Support for English and Lao scripts
* Export options for standard textgrid files
* Visual progress monitoring
* Lightweight interface for better usability

## 📁 File structure

The program expects a clear structure to function well. Place your audio files in one folder. Place your text files in another. Name the text files to match the audio files. For example, if you have recording_01.wav, ensure you have a file named recording_01.txt in the transcript folder.

## 🌐 Community and support

This project relies on open source libraries. These libraries provide the engine that makes the alignment possible. If you find bugs, check the repository provided above. Create a new issue if the program crashes repeatedly. Provide the version number and a description of your error to help others fix the problem.

## ℹ️ Licensing

The software remains free to use. You may share the software with others. Please credit the original authors when you build your own tools using this software.

Keywords: forced-alignment, lao, low-resource-languages, phonemes, pytorch, speech-processing, textgrid, tts, vits, wav2vec2