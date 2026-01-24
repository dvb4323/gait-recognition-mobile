/// Data model for a single sensor reading containing gyroscope and accelerometer data.
class SensorData {
  /// Gyroscope X-axis (rad/s)
  final double gx;
  
  /// Gyroscope Y-axis (rad/s)
  final double gy;
  
  /// Gyroscope Z-axis (rad/s)
  final double gz;
  
  /// Accelerometer X-axis (m/s²)
  final double ax;
  
  /// Accelerometer Y-axis (m/s²)
  final double ay;
  
  /// Accelerometer Z-axis (m/s²)
  final double az;
  
  /// Timestamp when the sample was taken
  final DateTime timestamp;

  SensorData({
    required this.gx,
    required this.gy,
    required this.gz,
    required this.ax,
    required this.ay,
    required this.az,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  /// Convert sensor data to list format: [Gx, Gy, Gz, Ax, Ay, Az]
  List<double> toList() {
    return [gx, gy, gz, ax, ay, az];
  }

  /// Create SensorData from a list [Gx, Gy, Gz, Ax, Ay, Az]
  factory SensorData.fromList(List<double> values, {DateTime? timestamp}) {
    if (values.length != 6) {
      throw ArgumentError('Expected 6 values, got ${values.length}');
    }
    return SensorData(
      gx: values[0],
      gy: values[1],
      gz: values[2],
      ax: values[3],
      ay: values[4],
      az: values[5],
      timestamp: timestamp,
    );
  }

  @override
  String toString() {
    return 'SensorData(gx: ${gx.toStringAsFixed(3)}, gy: ${gy.toStringAsFixed(3)}, '
        'gz: ${gz.toStringAsFixed(3)}, ax: ${ax.toStringAsFixed(3)}, '
        'ay: ${ay.toStringAsFixed(3)}, az: ${az.toStringAsFixed(3)})';
  }
}
