import 'dart:async';
import 'dart:math' as math;
import 'package:sensors_plus/sensors_plus.dart';
import 'package:vector_math/vector_math.dart';

/// Service for collecting real-time sensor data from accelerometer and gyroscope.
/// Implements gravity-aligned coordinate transformation for orientation-independent
/// gait recognition. Works regardless of how the phone is held.
class SensorService {
  /// Sampling rate in Hz (target)
  static const int samplingRate = 100;
  
  /// Window size in samples (2 seconds)
  static const int windowSize = 200;
  
  /// Low-pass filter coefficient for gravity estimation (0-1)
  /// Higher = more smoothing, slower response
  static const double _gravityFilterAlpha = 0.8;
  
  /// Gravitational acceleration for unit conversion (m/s² to g)
  static const double _g = 9.81;
  
  /// Buffer to accumulate sensor samples
  final List<List<double>> _buffer = [];
  
  /// Stream controller for complete windows
  final StreamController<List<List<double>>> _windowController = 
      StreamController<List<List<double>>>.broadcast();
  
  /// Stream subscriptions
  StreamSubscription? _gyroSubscription;
  StreamSubscription? _accelSubscription;
  
  /// Latest sensor values (for combining)
  List<double>? _latestGyro;
  List<double>? _latestAccel;
  
  /// Low-pass filtered gravity estimate (in g units)
  Vector3 _gravityEstimate = Vector3(0, 0, 1);
  
  /// Target gravity vector from training data (set via setTargetGravity)
  Vector3 _targetGravity = Vector3(0, -1, 0);
  
  /// Cached rotation matrix (updated when gravity estimate changes significantly)
  Matrix3 _rotationMatrix = Matrix3.identity();
  
  /// Whether collection is active
  bool _isCollecting = false;
  
  bool get isCollecting => _isCollecting;
  
  /// Current buffer size
  int get bufferSize => _buffer.length;
  
  /// Stream of complete windows ready for inference
  Stream<List<List<double>>> get windowStream => _windowController.stream;

  /// Set the target gravity vector from preprocessing params
  /// This determines how sensor data is transformed to match training data
  void setTargetGravity(Vector3 target) {
    _targetGravity = target.normalized();
    print('SensorService: Target gravity set to $_targetGravity');
  }

  /// Start collecting sensor data
  void startCollection() {
    if (_isCollecting) {
      print('SensorService: Already collecting');
      return;
    }
    
    _isCollecting = true;
    _buffer.clear();
    _latestGyro = null;
    _latestAccel = null;
    _gravityEstimate = Vector3(0, 0, 1); // Reset gravity estimate
    _rotationMatrix = Matrix3.identity();
    
    print('SensorService: Starting collection');
    print('  Target sampling rate: $samplingRate Hz');
    print('  Window size: $windowSize samples');
    print('  Target gravity: $_targetGravity');
    
    // Subscribe to gyroscope
    _gyroSubscription = gyroscopeEventStream(
      samplingPeriod: Duration(microseconds: (1000000 / samplingRate).round()),
    ).listen(
      (GyroscopeEvent event) {
        _latestGyro = [event.x, event.y, event.z];
        _tryAddSample();
      },
      onError: (error) {
        print('SensorService gyro error: $error');
      },
    );
    
    // Subscribe to accelerometer
    _accelSubscription = accelerometerEventStream(
      samplingPeriod: Duration(microseconds: (1000000 / samplingRate).round()),
    ).listen(
      (AccelerometerEvent event) {
        _latestAccel = [event.x, event.y, event.z];
        _tryAddSample();
      },
      onError: (error) {
        print('SensorService accel error: $error');
      },
    );
  }

  /// Update gravity estimate using low-pass filter
  void _updateGravityEstimate(Vector3 accel) {
    _gravityEstimate = Vector3(
      _gravityFilterAlpha * _gravityEstimate.x + (1 - _gravityFilterAlpha) * accel.x,
      _gravityFilterAlpha * _gravityEstimate.y + (1 - _gravityFilterAlpha) * accel.y,
      _gravityFilterAlpha * _gravityEstimate.z + (1 - _gravityFilterAlpha) * accel.z,
    );
  }

  /// Compute rotation matrix to align current gravity with target gravity
  /// Uses Rodrigues' rotation formula
  Matrix3 _computeRotationMatrix(Vector3 currentGravity, Vector3 targetGravity) {
    // Normalize vectors
    final current = currentGravity.normalized();
    final target = targetGravity.normalized();
    
    // Compute cross product (rotation axis)
    final axis = current.cross(target);
    final axisLength = axis.length;
    
    // If vectors are parallel, no rotation needed (or 180° flip)
    if (axisLength < 0.0001) {
      // Check if same direction or opposite
      if (current.dot(target) > 0) {
        return Matrix3.identity();
      } else {
        // 180° rotation around any perpendicular axis
        // Find a perpendicular vector
        Vector3 perp;
        if (current.x.abs() < 0.9) {
          perp = current.cross(Vector3(1, 0, 0)).normalized();
        } else {
          perp = current.cross(Vector3(0, 1, 0)).normalized();
        }
        // Rotation matrix for 180° around perp axis
        return _rodriguesRotation(perp, math.pi);
      }
    }
    
    // Normalize rotation axis
    final axisNorm = axis / axisLength;
    
    // Compute rotation angle
    final angle = math.acos(current.dot(target).clamp(-1.0, 1.0));
    
    return _rodriguesRotation(axisNorm, angle);
  }

  /// Rodrigues' rotation formula
  Matrix3 _rodriguesRotation(Vector3 axis, double angle) {
    final c = math.cos(angle);
    final s = math.sin(angle);
    final t = 1 - c;
    
    final x = axis.x;
    final y = axis.y;
    final z = axis.z;
    
    return Matrix3(
      t*x*x + c,     t*x*y - z*s,   t*x*z + y*s,
      t*x*y + z*s,   t*y*y + c,     t*y*z - x*s,
      t*x*z - y*s,   t*y*z + x*s,   t*z*z + c,
    );
  }

  /// Apply rotation matrix to a vector
  Vector3 _applyRotation(Matrix3 R, Vector3 v) {
    return Vector3(
      R.entry(0, 0) * v.x + R.entry(0, 1) * v.y + R.entry(0, 2) * v.z,
      R.entry(1, 0) * v.x + R.entry(1, 1) * v.y + R.entry(1, 2) * v.z,
      R.entry(2, 0) * v.x + R.entry(2, 1) * v.y + R.entry(2, 2) * v.z,
    );
  }

  /// Try to add a sample if both sensors have data
  void _tryAddSample() {
    if (_latestGyro == null || _latestAccel == null) {
      return;
    }
    
    // Convert accelerometer from m/s² to g units
    final accelG = Vector3(
      _latestAccel![0] / _g,
      _latestAccel![1] / _g,
      _latestAccel![2] / _g,
    );
    
    // Get gyroscope values (already in rad/s)
    final gyro = Vector3(_latestGyro![0], _latestGyro![1], _latestGyro![2]);
    
    // Update gravity estimate with low-pass filter
    _updateGravityEstimate(accelG);
    
    // Compute rotation matrix to align current gravity with target
    _rotationMatrix = _computeRotationMatrix(_gravityEstimate, _targetGravity);
    
    // Apply rotation to both accel and gyro
    final rotatedAccel = _applyRotation(_rotationMatrix, accelG);
    final rotatedGyro = _applyRotation(_rotationMatrix, gyro);
    
    // Create sample in order: [Gx, Gy, Gz, Ax, Ay, Az]
    final sample = [
      rotatedGyro.x,
      rotatedGyro.y,
      rotatedGyro.z,
      rotatedAccel.x,
      rotatedAccel.y,
      rotatedAccel.z,
    ];
    
    _buffer.add(sample);
    
    // Clear latest values to wait for next pair
    _latestGyro = null;
    _latestAccel = null;
    
    // Debug: Print buffer progress and gravity info every 50 samples
    if (_buffer.length % 50 == 0) {
      print('SensorService: Buffer ${_buffer.length}/$windowSize');
      if (_buffer.length == 50) {
        // Print debug info on first checkpoint
        print('  Current gravity estimate: $_gravityEstimate');
        print('  Transformed accel: $rotatedAccel');
      }
    }
    
    // When buffer reaches window size, emit and clear
    if (_buffer.length >= windowSize) {
      print('SensorService: Window complete (${_buffer.length} samples)');
      
      // Emit a copy of the buffer
      _windowController.add(List.from(_buffer));
      
      // Clear buffer for next window (no overlap)
      _buffer.clear();
    }
  }

  /// Stop collecting sensor data
  void stopCollection() {
    if (!_isCollecting) {
      print('SensorService: Not collecting');
      return;
    }
    
    _isCollecting = false;
    _gyroSubscription?.cancel();
    _accelSubscription?.cancel();
    _gyroSubscription = null;
    _accelSubscription = null;
    _buffer.clear();
    _latestGyro = null;
    _latestAccel = null;
    
    print('SensorService: Stopped collection');
  }

  /// Dispose resources
  void dispose() {
    stopCollection();
    _windowController.close();
    print('SensorService: Disposed');
  }
}
