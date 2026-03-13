package org.example;

import org.knowm.xchart.*;
import org.knowm.xchart.style.Styler;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.io.IOException;
import java.util.Map;

public class PlotResultsGated {

    public static void plotGatedResults(Map<Double, Double> psnrResults,
                                        Map<Double, Double> accResults,
                                        String datasetName) throws IOException {

        // Extract data
        double[] snr = psnrResults.keySet().stream().mapToDouble(Double::doubleValue).toArray();
        double[] psnr = psnrResults.values().stream().mapToDouble(Double::doubleValue).toArray();
        double[] acc = accResults.values().stream().mapToDouble(Double::doubleValue).toArray();

        // Create chart
        XYChart chart = new XYChartBuilder()
                .width(900)
                .height(600)
                .title("Gated Deep JSCC (Algorithm 2) - " + datasetName)
                .xAxisTitle("SNR (dB)")
                .yAxisTitle("PSNR (dB)")
                .build();

        // === Style setup ===
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNE);
        chart.getStyler().setLegendVisible(true);
        chart.getStyler().setMarkerSize(8);
        chart.getStyler().setChartTitleVisible(true);
        chart.getStyler().setDecimalPattern("#0.00");
        chart.getStyler().setAxisTitleFont(chart.getStyler().getBaseFont().deriveFont(16f));
        chart.getStyler().setChartTitleFont(chart.getStyler().getBaseFont().deriveFont(18f));
        chart.getStyler().setLegendFont(chart.getStyler().getBaseFont().deriveFont(14f));
        chart.getStyler().setXAxisDecimalPattern("#");
        chart.getStyler().setYAxisDecimalPattern("#0.0");
        chart.getStyler().setPlotGridVerticalLinesVisible(true);
        chart.getStyler().setPlotGridHorizontalLinesVisible(true);
        chart.getStyler().setPlotGridLinesVisible(true);
        chart.getStyler().setPlotBorderVisible(false);

        // PSNR curve
        XYSeries psnrSeries = chart.addSeries("PSNR (dB)", snr, psnr);
        psnrSeries.setMarker(SeriesMarkers.CIRCLE);
        psnrSeries.setLineWidth(3f);

        // Accuracy curve
        XYSeries accSeries = chart.addSeries("Accuracy (%)", snr, acc);
        accSeries.setMarker(SeriesMarkers.DIAMOND);
        accSeries.setLineWidth(3f);

        // Save chart
        String outputPath = "results/PSNR_Accuracy_vs_SNR_Gated_" + datasetName + ".png";
        BitmapEncoder.saveBitmap(chart, outputPath, BitmapEncoder.BitmapFormat.PNG);

        System.out.println("✅ Saved gated performance chart: " + outputPath);
    }
}
