import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:developer' as dv;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const YamNetVADApp());
}

class YamNetVADApp extends StatelessWidget {
  const YamNetVADApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YamNet VAD',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const VADHomePage(),
    );
  }
}

class VADHomePage extends StatefulWidget {
  const VADHomePage({super.key});

  @override
  _VADHomePageState createState() => _VADHomePageState();
}

class _VADHomePageState extends State<VADHomePage> {
  final _recorder = AudioRecorder();
  bool _isRecording = false;
  String _vadResult = 'No speech detected';
  String? _audioPath;
  Interpreter? _interpreter;
  Timer? _vadTimer;
  int _segmentCounter = 0;

  @override
  void initState() {
    super.initState();
    _initTFLite();
    _requestPermissions();
  }

  Future<void> _initTFLite() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/vad.tflite');
      dv.log('TFLite model loaded successfully');
      var inputTensor = _interpreter!.getInputTensor(0);
      var outputTensor = _interpreter!.getOutputTensor(0);
      dv.log('Input tensor shape: ${inputTensor.shape}');
      print('Output tensor shape: ${outputTensor.shape}');
      if (inputTensor.shape.length != 2 || outputTensor.shape.length != 2) {
        dv.log('Warning: Model has non-2D tensor shapes');
        setState(() {
          _vadResult = 'Invalid model tensor shapes';
        });
      }
    } catch (e) {
      dv.log('Error loading TFLite model: $e');
      setState(() {
        _vadResult = 'Failed to load model';
      });
    }
  }

  Future<void> _requestPermissions() async {
    if (await Permission.microphone.request().isGranted) {
      dv.log('Microphone permission granted');
    } else {
      dv.log('Microphone permission denied');
      setState(() {
        _vadResult = 'Microphone permission denied';
      });
    }
  }

  Future<void> _startRecording() async {
    if (await _recorder.hasPermission()) {
      final directory = await getTemporaryDirectory();
      _audioPath = '${directory.path}/recording_$_segmentCounter.wav';
      await _recorder.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          bitRate: 128000,
          sampleRate: 16000, // YamNet expects 16kHz audio
          numChannels: 1, // Mono
        ),
        path: _audioPath!,
      );
      setState(() {
        _isRecording = true;
        _vadResult = 'Recording...';
      });
      Future.delayed(const Duration(milliseconds: 1500), () {
        if (_isRecording) _startVADProcessing();
      });
    }
  }

  Future<void> _stopRecording() async {
    await _recorder.stop();
    _vadTimer?.cancel();
    _segmentCounter = 0;
    setState(() {
      _isRecording = false;
      _vadResult = 'No speech detected';
    });
  }

  void _startVADProcessing() {
    _vadTimer = Timer.periodic(const Duration(seconds: 1), (timer) async {
      if (_isRecording && _audioPath != null && _interpreter != null) {
        _segmentCounter++;
        final directory = await getTemporaryDirectory();
        final newAudioPath = '${directory.path}/recording_$_segmentCounter.wav';
        await _recorder.stop();
        await _recorder.start(
          const RecordConfig(
            encoder: AudioEncoder.wav,
            bitRate: 128000,
            sampleRate: 16000,
            numChannels: 1,
          ),
          path: newAudioPath,
        );
        await _processAudioForVAD(_audioPath!);
        _audioPath = newAudioPath;
      }
    });
  }

  Future<void> _processAudioForVAD(String audioPath) async {
    try {
      // Read the recorded audio file
      File audioFile = File(audioPath);
      if (!await audioFile.exists() || await audioFile.length() < 44) {
        dv.log('Audio file is missing or too small: $audioPath');
        return;
      }

      // Preprocess audio
      Uint8List audioBytes = await audioFile.readAsBytes();
      dv.log('Audio file size: ${audioBytes.length} bytes');
      Float32List audioInput = _preprocessAudio(audioBytes);
      if (audioInput.isEmpty) {
        dv.log('No valid audio data after preprocessing');
        return;
      }
      dv.log(
          'First 10 audio samples: ${audioInput.sublist(0, min(10, audioInput.length))}');

      // Prepare input tensor ([1, 15600] float32)
      var inputTensor = _interpreter!.getInputTensor(0);
      var inputShape =
          inputTensor.shape.isNotEmpty && inputTensor.shape.length == 2
              ? inputTensor.shape
              : [1, 15600];
      dv.log('Input tensor shape: $inputShape');
      if (inputShape.length != 2) {
        dv.log('Invalid input tensor shape: $inputShape');
        setState(() {
          _vadResult = 'Invalid model input shape';
        });
        return;
      }
      var input = ListExtension(List.filled(inputShape[0] * inputShape[1], 0.0))
          .reshape(inputShape);
      for (int i = 0; i < audioInput.length && i < inputShape[1]; i++) {
        input[0][i] = audioInput[i];
      }

      // Prepare output tensor ([1, 521] float32)
      var outputTensor = _interpreter!.getOutputTensor(0);
      var outputShape =
          outputTensor.shape.isNotEmpty && outputTensor.shape.length == 2
              ? outputTensor.shape
              : [1, 521];
      dv.log('Output tensor shape: $outputShape');
      if (outputShape.length != 2) {
        dv.log('Invalid output tensor shape: $outputShape');
        setState(() {
          _vadResult = 'Invalid model output shape';
        });
        return;
      }
      var output =
          ListExtension(List.filled(outputShape[0] * outputShape[1], 0.0))
              .reshape(outputShape);

      // Run inference
      _interpreter!.run(input, output);

      // Apply sigmoid to convert logits to probabilities
      List<double> probabilities = (output[0] as List)
          .map((score) => 1 / (1 + exp(-(score as double))))
          .toList()
          .cast<double>();

      // Log top 5 classes
      List<MapEntry<int, double>> indexedScores = probabilities
          .asMap()
          .entries
          .toList()
        ..sort((a, b) => b.value.compareTo(a.value));
      dv.log('Top 5 class scores:');
      for (int i = 0; i < 5 && i < indexedScores.length; i++) {
        dv.log('Class ${indexedScores[i].key}: ${indexedScores[i].value}');
      }

      // Check for speech (class index 0)
      bool speechDetected = probabilities[0] > 0.69; // Lowered threshold
      dv.log('Speech class score (probability): ${probabilities[0]}');
      setState(() {
        _vadResult = speechDetected ? 'Speech detected!' : 'No speech detected';
      });
    } catch (e) {
      dv.log('Error processing VAD: $e');
      setState(() {
        _vadResult = 'Error processing audio';
      });
    }
  }

  Float32List _preprocessAudio(Uint8List audioBytes) {
    // Convert WAV bytes to float32 in [-1.0, 1.0]
    const headerSize = 44;
    if (audioBytes.length <= headerSize) {
      dv.log('Audio data too short: ${audioBytes.length} bytes');
      return Float32List(0);
    }

    // Convert 16-bit PCM to float32
    Int16List pcm;
    try {
      pcm = Int16List.view(audioBytes.buffer, headerSize);
    } catch (e) {
      dv.log('Error parsing PCM data: $e');
      return Float32List(0);
    }
    dv.log('PCM samples: ${pcm.length}');

    // Take the last 15600 samples (~0.975s at 16kHz)
    const targetSamples = 15600;
    int startIndex =
        pcm.length > targetSamples ? pcm.length - targetSamples : 0;
    int length = pcm.length > targetSamples ? targetSamples : pcm.length;
    Float32List audioInput = Float32List(targetSamples);
    audioInput.fillRange(0, targetSamples, 0.0);
    for (int i = 0; i < length; i++) {
      audioInput[i] = pcm[startIndex + i] / 32768.0; // Normalize to [-1.0, 1.0]
    }
    return audioInput;
  }

  @override
  void dispose() {
    _vadTimer?.cancel();
    _recorder.dispose();
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('YamNet Voice Activity Detection'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              _vadResult,
              style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _isRecording ? _stopRecording : _startRecording,
              child: Text(_isRecording ? 'Stop Recording' : 'Start Recording'),
            ),
          ],
        ),
      ),
    );
  }
}

extension ListExtension on List<double> {
  List<List<double>> reshape(List<int> shape) {
    if (shape.length != 2) {
      dv.log('Invalid shape for reshape: $shape');
      throw Exception('Only 2D reshaping supported');
    }
    List<List<double>> result =
        List.generate(shape[0], (_) => List.filled(shape[1], 0.0));
    int index = 0;
    for (int i = 0; i < shape[0] && index < length; i++) {
      for (int j = 0; j < shape[1] && index < length; j++) {
        result[i][j] = this[index++];
      }
    }
    return result;
  }

  int reduce(int Function(int, dynamic) combine) {
    return fold(1, combine);
  }
}
