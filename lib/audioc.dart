import 'dart:async';
import 'dart:typed_data';
import 'package:record/record.dart';
import 'package:permission_handler/permission_handler.dart';

class MAudioRecorder {
  final _recorder = AudioRecorder();
  StreamController<Float32List>? _audioStreamController;
  bool _isRecording = false;
  static const double _gain = 5.0; // Increased gain to boost low-volume audio
  static const int _samplesPerChunk =
      15600; // YAMNet requires 15,600 samples (0.975s at 16 kHz)

  Future<void> init() async {
    // No explicit initialization needed for the `record` package
    print('AudioRecorder initialized');
  }

  Future<bool> requestPermission() async {
    var status = await Permission.microphone.request();
    if (status.isGranted) {
      print('Microphone permission granted');
    } else {
      print('Microphone permission denied');
    }
    return status.isGranted;
  }

  Stream<Float32List> startRecording() async* {
    _audioStreamController = StreamController<Float32List>();
    _isRecording = true;

    // Configure recording settings
    const recordConfig = RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: 16000,
      numChannels: 1,
    );

    // Check if recording is supported
    if (await _recorder.hasPermission()) {
      print('Starting audio stream with config: $recordConfig');
      // Start recording and stream PCM data
      final stream = await _recorder.startStream(recordConfig);

      // Buffer to collect 15,600 samples
      final buffer = <int>[];

      stream.listen(
        (data) {
          if (!_isRecording) return;

          // Add incoming PCM16 data to buffer
          final int16Buffer = Int16List.view(data.buffer);
          print(
              'Received PCM chunk: ${int16Buffer.length} samples, total buffered: ${buffer.length + int16Buffer.length}');
          buffer.addAll(int16Buffer);

          // Process when buffer has at least 15,600 samples
          while (buffer.length >= _samplesPerChunk) {
            final chunk = buffer.sublist(0, _samplesPerChunk);
            buffer.removeRange(0, _samplesPerChunk);

            // Convert PCM16 to Float32 for YAMNet with gain
            final float32List = Float32List(_samplesPerChunk);
            for (int i = 0; i < _samplesPerChunk; i++) {
              // Normalize to [-1, 1] and apply gain
              double sample = (chunk[i] / 32768.0) * _gain;
              // Clip to [-1, 1] to prevent overflow
              float32List[i] = sample.clamp(-1.0, 1.0);
            }

            // Log audio chunk statistics
            double minVal = float32List.reduce((a, b) => a < b ? a : b);
            double maxVal = float32List.reduce((a, b) => a > b ? a : b);
            double meanVal = float32List.fold(0.0, (sum, val) => sum + val) /
                float32List.length;
            print(
                'Processed chunk: min=$minVal, max=$maxVal, mean=$meanVal, length=${float32List.length}');

            _audioStreamController!.add(float32List);
          }
        },
        onError: (e) {
          print('Recording error: $e');
          _audioStreamController?.addError(e);
        },
        onDone: () {
          print('Recording stream closed');
          _audioStreamController?.close();
        },
      );
    } else {
      print('Recording permission not granted');
      _audioStreamController?.addError('Recording permission not granted');
    }

    yield* _audioStreamController!.stream;
  }

  Future<void> stopRecording() async {
    if (_isRecording) {
      print('Stopping recording');
      await _recorder.stop();
      await _audioStreamController?.close();
      _isRecording = false;
    }
  }

  Future<void> dispose() async {
    print('Disposing AudioRecorder');
    await stopRecording();
    await _recorder.dispose();
  }
}
