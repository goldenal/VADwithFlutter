import 'dart:async';
import 'dart:developer';
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';

class VADService {
  Interpreter? _interpreter;
  static const int _speechClassIndex =
      0; // YAMNet's "Speech" class is at index 0
  static const int _numClasses = 521; // YAMNet has 521 classes
  static const double _speechThreshold =
      0.5; // Confidence threshold for speech detection

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/vad.tflite');
      log('YAMNet model loaded successfully');
    } catch (e) {
      print('Error loading model: $e');
    }
  }

  Future<bool> detectVoice(Float32List audioData) async {
    if (_interpreter == null) {
      log('Model not loaded');
      return false;
    }
    // Log input audio statistics
    double minVal = audioData.reduce((a, b) => a < b ? a : b);
    double maxVal = audioData.reduce((a, b) => a > b ? a : b);
    double meanVal =
        audioData.fold(0.0, (sum, val) => sum + val) / audioData.length;
    log('Audio input stats: min=$minVal, max=$maxVal, mean=$meanVal, length=${audioData.length}');

    // Prepare input tensor (YAMNet expects 15,600 samples at 16 kHz)
    var input = audioData.reshape([1, 15600]);
    var output = List.filled(1 * _numClasses, 0.0).reshape([1, _numClasses]);

    // Run inferences
    _interpreter!.run(input, output);

    // Find top 5 class scores
    List<double> scores = output[0].cast<double>();
    List<int> indices = List.generate(_numClasses, (i) => i);
    indices.sort(
        (a, b) => scores[b].compareTo(scores[a])); // Sort by score descending
    log('Top 5 classes:');
    for (int i = 0; i < 5 && i < indices.length; i++) {
      log('Class ${indices[i]}: ${scores[indices[i]]} ${indices[i] == _speechClassIndex ? "(Speech)" : ""}');
    }

    // Check the score for the "Speech" class (index 0)
    double speechScore = output[0][_speechClassIndex];
    log('Speech score: $speechScore');

    return speechScore > _speechThreshold;
  }

  void dispose() {
    _interpreter?.close();
  }
}
