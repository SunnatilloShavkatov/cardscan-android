package com.getbouncer.scan.payment.analyzer

import kotlin.test.Test
import kotlin.test.assertEquals

class ExpiryOcrAnalyzerTest {
    @Test
    fun parseSlashSeparatedExpiry() {
        val result = ExpiryOcrAnalyzer.ExpiryParser.parse("VALID THRU 03/29")

        assertEquals("03", result.single().month)
        assertEquals("2029", result.single().year)
    }

    @Test
    fun parseSpaceSeparatedExpiry() {
        val result = ExpiryOcrAnalyzer.ExpiryParser.parse("MONTH YEAR 11 2030")

        assertEquals("11", result.single().month)
        assertEquals("2030", result.single().year)
    }

    @Test
    fun parseCompactExpiry() {
        val result = ExpiryOcrAnalyzer.ExpiryParser.parse("0728")

        assertEquals("07", result.single().month)
        assertEquals("2028", result.single().year)
    }
}
