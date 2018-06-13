# Subscribe /binaural_audio/source_stream', and publish to 'acoustic_anomaly' and 'chatbot/output' for acoustic anomaly detection.

**Usage:**
### Launch acoustic_anomaly:
$roslaunch acoustic_anomaly publish_acoustic_anomaly.launch
### Visualize in rqt
$roslaunch acoustic_anomaly visualize_acoustic_anomaly.launch


**Requirements:**
1. keras==2.0.7
2. tensorflow==1.3
3. h5py==2.7.1
4. scikit-learn==0.19.1
5. python_speech_features==0.5
