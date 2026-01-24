import 'dart:async';
import 'package:flutter/material.dart';
import '../models/prediction_result.dart';
import '../services/sensor_service.dart';
import '../services/preprocessing_service.dart';
import '../services/native_model_service.dart';
import '../widgets/activity_indicator.dart';
import '../widgets/confidence_chart.dart';

/// Main home screen for gait activity classification.
/// Displays real-time activity predictions from sensor data.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // Services
  final SensorService _sensorService = SensorService();
  final PreprocessingService _preprocessingService = PreprocessingService();
  final NativeModelService _modelService = NativeModelService();
  
  // State
  bool _isInitialized = false;
  bool _isLoading = true;
  bool _isRunning = false;
  String? _errorMessage;
  PredictionResult? _currentPrediction;
  int _predictionCount = 0;
  int _bufferSize = 0;
  
  // Stream subscription
  StreamSubscription? _windowSubscription;
  
  // Timer for UI updates
  Timer? _uiUpdateTimer;

  @override
  void initState() {
    super.initState();
    _initializeServices();
  }

  Future<void> _initializeServices() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });
    
    try {
      // Load preprocessing parameters
      print('HomeScreen: Loading preprocessing parameters...');
      await _preprocessingService.loadParams();
      
      // Set target gravity on sensor service for orientation-independent transformation
      _sensorService.setTargetGravity(_preprocessingService.targetGravity);
      
      // Load TFLite model via native Platform Channel
      print('HomeScreen: Loading TFLite model (native)...');
      await _modelService.loadModel();
      
      // Setup window stream listener
      _windowSubscription = _sensorService.windowStream.listen(_processWindow);
      
      setState(() {
        _isInitialized = true;
        _isLoading = false;
      });
      
      print('HomeScreen: Initialization complete!');
    } catch (e) {
      print('HomeScreen: Initialization error: $e');
      setState(() {
        _errorMessage = 'Failed to initialize: $e';
        _isLoading = false;
      });
    }
  }

  Future<void> _processWindow(List<List<double>> window) async {
    if (!_isRunning) return;
    
    try {
      // Debug: Log raw sensor data stats for first prediction
      if (_predictionCount == 0) {
        // Calculate mean of each feature in raw window
        List<double> sums = List.filled(6, 0.0);
        for (var sample in window) {
          for (int i = 0; i < 6; i++) {
            sums[i] += sample[i];
          }
        }
        List<double> means = sums.map((s) => s / window.length).toList();
        print('DEBUG: Raw sensor data means [Gx,Gy,Gz,Ax,Ay,Az]: $means');
        print('DEBUG: Expected means from params: ${_preprocessingService.mean}');
        print('DEBUG: First sample: ${window[0]}');
      }
      
      // Preprocess the window
      final input = _preprocessingService.prepareForInference(window);
      
      // Run inference via native Platform Channel (async)
      final result = await _modelService.predict(input);
      
      // Update UI
      if (mounted) {
        setState(() {
          _currentPrediction = result;
          _predictionCount++;
        });
      }
      
      print('HomeScreen: Prediction #$_predictionCount: ${result.activityLabel} (${result.confidencePercent})');
    } catch (e) {
      print('HomeScreen: Prediction error: $e');
    }
  }

  void _toggleRecording() {
    setState(() {
      _isRunning = !_isRunning;
      
      if (_isRunning) {
        _sensorService.startCollection();
        _predictionCount = 0;
        _bufferSize = 0;
        
        // Start timer to update UI with buffer progress
        _uiUpdateTimer = Timer.periodic(
          const Duration(milliseconds: 200),
          (_) {
            if (mounted && _isRunning) {
              setState(() {
                _bufferSize = _sensorService.bufferSize;
              });
            }
          },
        );
      } else {
        _sensorService.stopCollection();
        _uiUpdateTimer?.cancel();
        _uiUpdateTimer = null;
        _bufferSize = 0;
      }
    });
  }

  @override
  void dispose() {
    _uiUpdateTimer?.cancel();
    _windowSubscription?.cancel();
    _sensorService.dispose();
    _modelService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Gait Recognition'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          // Show prediction count when running
          if (_isRunning)
            Center(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Text(
                  'Predictions: $_predictionCount',
                  style: const TextStyle(fontSize: 14),
                ),
              ),
            ),
        ],
      ),
      body: _buildBody(),
      floatingActionButton: _isInitialized
          ? FloatingActionButton.extended(
              onPressed: _toggleRecording,
              icon: Icon(_isRunning ? Icons.stop : Icons.play_arrow),
              label: Text(_isRunning ? 'Stop' : 'Start'),
              backgroundColor: _isRunning ? Colors.red : Colors.green,
              foregroundColor: Colors.white,
            )
          : null,
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }

  Widget _buildBody() {
    // Loading state
    if (_isLoading) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Loading model...'),
          ],
        ),
      );
    }
    
    // Error state
    if (_errorMessage != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, size: 64, color: Colors.red),
              const SizedBox(height: 16),
              Text(
                _errorMessage!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.red),
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: _initializeServices,
                child: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }
    
    // Main content
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Status indicator
          _buildStatusCard(),
          const SizedBox(height: 24),
          
          // Activity indicator
          ActivityIndicator(
            prediction: _currentPrediction,
            isActive: _isRunning,
          ),
          const SizedBox(height: 24),
          
          // Probability chart
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Probability Distribution',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 16),
                  SizedBox(
                    height: 200,
                    child: ConfidenceChart(prediction: _currentPrediction),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),
          
          // Instructions
          _buildInstructionsCard(),
          
          // Bottom padding for FAB
          const SizedBox(height: 80),
        ],
      ),
    );
  }

  Widget _buildStatusCard() {
    final Color statusColor = _isRunning ? Colors.green : Colors.grey;
    final String statusText = _isRunning 
        ? 'Collecting sensor data...' 
        : 'Press Start to begin';
    
    return Card(
      color: statusColor.withOpacity(0.1),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Container(
              width: 12,
              height: 12,
              decoration: BoxDecoration(
                color: statusColor,
                shape: BoxShape.circle,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                statusText,
                style: TextStyle(
                  color: statusColor,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
            if (_isRunning)
              Text(
                'Buffer: $_bufferSize/200',
                style: TextStyle(
                  color: statusColor.withOpacity(0.7),
                  fontSize: 12,
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildInstructionsCard() {
    return Card(
      color: Colors.blue.withOpacity(0.1),
      child: const Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.info_outline, color: Colors.blue, size: 20),
                SizedBox(width: 8),
                Text(
                  'Instructions',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: Colors.blue,
                  ),
                ),
              ],
            ),
            SizedBox(height: 12),
            Text('1. Place phone at waist/pocket (center position)'),
            Text('2. Press Start to begin classification'),
            Text('3. Walk normally - predictions update every 2 seconds'),
            Text('4. Press Stop to end session'),
          ],
        ),
      ),
    );
  }
}
