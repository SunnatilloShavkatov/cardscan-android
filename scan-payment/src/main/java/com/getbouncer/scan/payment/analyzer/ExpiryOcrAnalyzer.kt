package com.getbouncer.scan.payment.analyzer

import android.graphics.Bitmap
import android.graphics.Rect
import com.getbouncer.scan.framework.Analyzer
import com.getbouncer.scan.framework.AnalyzerFactory
import com.getbouncer.scan.framework.TrackedImage
import com.getbouncer.scan.payment.cropCameraPreviewToSquare
import com.getbouncer.scan.payment.ml.ExpiryDetect
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.tasks.await
import java.io.Closeable

private val EXPIRY_WITH_SEPARATOR_REGEX =
    Regex("""(?<!\d)(0?[1-9]|1[0-2])\s*[/\\|.\-]\s*(\d{2}|\d{4})(?!\d)""")
private val EXPIRY_WITH_SPACE_REGEX =
    Regex("""(?<!\d)(0?[1-9]|1[0-2])\s+(\d{2}|\d{4})(?!\d)""")
private val EXPIRY_COMPACT_REGEX =
    Regex("""(?<!\d)(0[1-9]|1[0-2])(\d{2})(?!\d)""")

@Deprecated(
    message = "Replaced by stripe card scan. See https://github.com/stripe/stripe-android/tree/master/stripecardscan",
    replaceWith = ReplaceWith("StripeCardScan"),
)
class ExpiryOcrAnalyzer private constructor(
    private val textRecognizer: TextRecognizer,
) : Analyzer<ExpiryOcrAnalyzer.Input, Any, ExpiryOcrAnalyzer.Prediction>, Closeable {

    @Deprecated(
        message = "Replaced by stripe card scan. See https://github.com/stripe/stripe-android/tree/master/stripecardscan",
        replaceWith = ReplaceWith("StripeCardScan"),
    )
    data class Input(
        val cameraPreviewImage: TrackedImage<Bitmap>,
        val previewBounds: Rect,
        val cardFinder: Rect,
    )

    @Deprecated(
        message = "Replaced by stripe card scan. See https://github.com/stripe/stripe-android/tree/master/stripecardscan",
        replaceWith = ReplaceWith("StripeCardScan"),
    )
    data class Prediction(
        val expiry: ExpiryDetect.Expiry?,
    )

    private data class CropSpec(
        val leftRatio: Float,
        val topRatio: Float,
        val rightRatio: Float,
        val bottomRatio: Float,
        val baseScore: Int,
    )

    private data class ExpiryCandidate(
        val expiry: ExpiryDetect.Expiry,
        val score: Int,
    )

    override suspend fun analyze(data: Input, state: Any): Prediction {
        val squareImage = cropCameraPreviewToSquare(
            cameraPreviewImage = data.cameraPreviewImage.image,
            previewBounds = data.previewBounds,
            viewFinder = data.cardFinder,
        )

        val candidates = mutableListOf<ExpiryCandidate>()
        for (cropSpec in CROP_SPECS) {
            val croppedBitmap = cropBitmap(squareImage, cropSpec) ?: continue
            val recognizedText = textRecognizer.process(InputImage.fromBitmap(croppedBitmap, 0)).await()
            candidates.addAll(extractCandidates(croppedBitmap, recognizedText, cropSpec.baseScore))
        }

        return Prediction(candidates.maxByOrNull { it.score }?.expiry)
    }

    override fun close() {
        textRecognizer.close()
    }

    private fun extractCandidates(
        bitmap: Bitmap,
        recognizedText: Text,
        baseScore: Int,
    ): List<ExpiryCandidate> {
        val candidates = mutableListOf<ExpiryCandidate>()

        for (block in recognizedText.textBlocks) {
            for (line in block.lines) {
                val lineScore = baseScore + scoreLine(bitmap, line)
                for (expiry in ExpiryParser.parse(line.text)) {
                    candidates.add(ExpiryCandidate(expiry, lineScore))
                }
            }
        }

        val fullTextScore = baseScore + scoreText(recognizedText.text) - 10
        for (expiry in ExpiryParser.parse(recognizedText.text)) {
            candidates.add(ExpiryCandidate(expiry, fullTextScore))
        }

        return candidates
    }

    private fun scoreLine(bitmap: Bitmap, line: Text.Line): Int {
        val box = line.boundingBox
        val verticalBias = if (box != null && bitmap.height > 0) {
            (box.centerY().toFloat() / bitmap.height.toFloat() * 25f).toInt()
        } else {
            0
        }
        return scoreText(line.text) + verticalBias
    }

    private fun scoreText(text: String): Int {
        var score = 0
        val upper = text.uppercase()
        if (upper.contains('/')) score += 30
        if (upper.contains("VALID") || upper.contains("THRU") || upper.contains("EXP")) score += 20
        if (upper.contains("MONTH") || upper.contains("YEAR")) score += 15
        if (text.length <= 12) score += 10
        if (text.count { it.isDigit() } > 10) score -= 20
        return score
    }

    private fun cropBitmap(bitmap: Bitmap, cropSpec: CropSpec): Bitmap? {
        val left = (bitmap.width * cropSpec.leftRatio).toInt().coerceIn(0, bitmap.width)
        val top = (bitmap.height * cropSpec.topRatio).toInt().coerceIn(0, bitmap.height)
        val right = (bitmap.width * cropSpec.rightRatio).toInt().coerceIn(left, bitmap.width)
        val bottom = (bitmap.height * cropSpec.bottomRatio).toInt().coerceIn(top, bitmap.height)
        val width = right - left
        val height = bottom - top
        return if (width > 0 && height > 0) {
            Bitmap.createBitmap(bitmap, left, top, width, height)
        } else {
            null
        }
    }

    @Deprecated(
        message = "Replaced by stripe card scan. See https://github.com/stripe/stripe-android/tree/master/stripecardscan",
        replaceWith = ReplaceWith("StripeCardScan"),
    )
    class Factory : AnalyzerFactory<Input, Any, Prediction, ExpiryOcrAnalyzer> {
        override suspend fun newInstance() = ExpiryOcrAnalyzer(
            textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS),
        )
    }

    internal object ExpiryParser {
        fun parse(text: String?): List<ExpiryDetect.Expiry> {
            if (text.isNullOrBlank()) {
                return emptyList()
            }

            val normalized = normalize(text)
            val candidates = linkedSetOf<ExpiryDetect.Expiry>()

            addMatches(normalized, EXPIRY_WITH_SEPARATOR_REGEX, candidates)
            addMatches(normalized, EXPIRY_WITH_SPACE_REGEX, candidates)

            val compactText = normalized.replace(" ", "")
            if (compactText.length <= 6) {
                addMatches(compactText, EXPIRY_COMPACT_REGEX, candidates)
            }

            return candidates.toList()
        }

        private fun addMatches(
            text: String,
            regex: Regex,
            output: MutableSet<ExpiryDetect.Expiry>,
        ) {
            for (match in regex.findAll(text)) {
                val expiry = createExpiry(
                    month = match.groupValues[1],
                    year = match.groupValues[2],
                ) ?: continue
                output.add(expiry)
            }
        }

        private fun createExpiry(
            month: String,
            year: String,
        ): ExpiryDetect.Expiry? {
            val normalizedMonth = month.padStart(2, '0').takeLast(2)
            val normalizedYear = if (year.length == 2) {
                "20$year"
            } else {
                year.takeLast(4)
            }
            val expiry = ExpiryDetect.Expiry(normalizedMonth, normalizedYear)
            return expiry.takeIf { it.isValidExpiry() }
        }

        private fun normalize(text: String): String = buildString(text.length) {
            for (char in text.uppercase()) {
                append(
                    when (char) {
                        'O', 'Q', 'D' -> '0'
                        'I', 'L' -> '1'
                        'Z' -> '2'
                        'S' -> '5'
                        'B' -> '8'
                        else -> char
                    }
                )
            }
        }
            .replace(Regex("[^0-9A-Z/\\\\|.\\- ]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
    }

    companion object {
        private val CROP_SPECS = listOf(
            CropSpec(0f, 0f, 1f, 1f, 0),
            CropSpec(0f, 0.45f, 1f, 1f, 20),
            CropSpec(0.2f, 0.45f, 0.9f, 0.82f, 25),
            CropSpec(0.35f, 0.35f, 1f, 0.9f, 15),
        )
    }
}
