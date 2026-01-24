package com.example.gait_recognition_flutter

import android.content.res.AssetManager
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity: FlutterActivity() {
    private val CHANNEL = "com.example.gait_recognition_flutter/tflite"
    private var interpreter: Interpreter? = null
    private var isModelLoaded = false
    private var currentModelPath: String? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "loadModel" -> {
                    val modelPath = call.argument<String>("modelPath") ?: "gait_lstm_model.tflite"
                    try {
                        loadModel(modelPath)
                        result.success(mapOf(
                            "success" to true,
                            "inputShape" to interpreter?.getInputTensor(0)?.shape()?.toList(),
                            "outputShape" to interpreter?.getOutputTensor(0)?.shape()?.toList()
                        ))
                    } catch (e: Exception) {
                        result.error("LOAD_ERROR", "Failed to load model: ${e.message}", null)
                    }
                }
                "predict" -> {
                    val inputData = call.argument<List<List<List<Double>>>>("input")
                    if (inputData == null) {
                        result.error("INVALID_INPUT", "Input data is null", null)
                        return@setMethodCallHandler
                    }
                    try {
                        val output = runInference(inputData)
                        result.success(mapOf(
                            "success" to true,
                            "probabilities" to output
                        ))
                    } catch (e: Exception) {
                        result.error("INFERENCE_ERROR", "Inference failed: ${e.message}", null)
                    }
                }
                "closeModel" -> {
                    closeModel()
                    result.success(true)
                }
                "isModelLoaded" -> {
                    result.success(isModelLoaded)
                }
                else -> {
                    result.notImplemented()
                }
            }
        }
    }

    private fun loadModel(modelPath: String) {
        // Close existing interpreter if any
        closeModel()
        
        // Load model from Flutter assets
        // Flutter assets are located at flutter_assets/assets/... in native Android
        val assetPath = "flutter_assets/assets/models/$modelPath"
        val modelBuffer = loadModelFile(assets, assetPath)
        android.util.Log.d("TFLite", "Loading model from: $assetPath")
        
        // Create interpreter options with Flex delegate for SELECT_TF_OPS
        val options = Interpreter.Options()
        
        // Add Flex delegate for GRU/LSTM support
        try {
            val flexDelegate = FlexDelegate()
            options.addDelegate(flexDelegate)
            android.util.Log.d("TFLite", "Flex delegate added successfully")
        } catch (e: Exception) {
            android.util.Log.w("TFLite", "Could not add Flex delegate: ${e.message}")
        }
        
        // Set number of threads
        options.setNumThreads(4)
        
        // Create interpreter
        interpreter = Interpreter(modelBuffer, options)
        isModelLoaded = true
        currentModelPath = modelPath
        
        android.util.Log.d("TFLite", "Model loaded: $modelPath")
        android.util.Log.d("TFLite", "Input shape: ${interpreter?.getInputTensor(0)?.shape()?.contentToString()}")
        android.util.Log.d("TFLite", "Output shape: ${interpreter?.getOutputTensor(0)?.shape()?.contentToString()}")
    }

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun runInference(inputData: List<List<List<Double>>>): List<Float> {
        if (interpreter == null || !isModelLoaded) {
            throw IllegalStateException("Model not loaded")
        }
        
        // Input shape: [1, 200, 6]
        val batchSize = inputData.size
        val timeSteps = inputData[0].size
        val features = inputData[0][0].size
        
        // Create input buffer
        val inputBuffer = ByteBuffer.allocateDirect(batchSize * timeSteps * features * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        // Fill input buffer
        for (batch in inputData) {
            for (timeStep in batch) {
                for (feature in timeStep) {
                    inputBuffer.putFloat(feature.toFloat())
                }
            }
        }
        inputBuffer.rewind()
        
        // Create output buffer: [1, 5]
        val outputBuffer = ByteBuffer.allocateDirect(batchSize * 5 * 4)
        outputBuffer.order(ByteOrder.nativeOrder())
        
        // Run inference
        interpreter?.run(inputBuffer, outputBuffer)
        
        // Extract results
        outputBuffer.rewind()
        val results = mutableListOf<Float>()
        for (i in 0 until 5) {
            results.add(outputBuffer.getFloat())
        }
        
        return results
    }

    private fun closeModel() {
        interpreter?.close()
        interpreter = null
        isModelLoaded = false
        currentModelPath = null
    }

    override fun onDestroy() {
        closeModel()
        super.onDestroy()
    }
}
