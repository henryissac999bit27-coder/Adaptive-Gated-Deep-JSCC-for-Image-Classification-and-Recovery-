package org.example;

import org.knowm.xchart.*;
import org.knowm.xchart.style.markers.SeriesMarkers;
import java.util.*;

public class PlotResults {

    public static void plotPSNRandAccuracy(Map<Double, Double> psnrMap, Map<Double, Double> accMap, String datasetName) {

        // Convert maps to sorted lists
        List<Double> snrValues = new ArrayList<>(psnrMap.keySet());
        Collections.sort(snrValues);
        List<Double> psnrValues = new ArrayList<>();
        List<Double> accValues = new ArrayList<>();
        for (double snr : snrValues) {
            psnrValues.add(psnrMap.get(snr));
            accValues.add(accMap.get(snr));
        }

        // Create chart
        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(500)
                .title(datasetName + " - PSNR and Accuracy vs SNR")
                .xAxisTitle("SNR (dB)")
                .yAxisTitle("PSNR (dB)")
                .build();

        // PSNR series (left Y-axis)
        XYSeries psnrSeries = chart.addSeries("PSNR (dB)", snrValues, psnrValues);
        psnrSeries.setMarker(SeriesMarkers.SQUARE);

        // Accuracy series (right Y-axis)
        XYSeries accSeries = chart.addSeries("Classification Accuracy", snrValues, accValues);
        accSeries.setMarker(SeriesMarkers.TRIANGLE_UP);
        accSeries.setYAxisGroup(1);

        // Add second axis for accuracy
        chart.setYAxisGroupTitle(1, "Accuracy (%)");

        // Display chart
        new SwingWrapper<>(chart).displayChart();

        // Optionally save
        try {
            BitmapEncoder.saveBitmap(chart, "results/PSNR_Accuracy_vs_SNR_" + datasetName, BitmapEncoder.BitmapFormat.PNG);
            System.out.println("✅ Saved chart: results/PSNR_Accuracy_vs_SNR_" + datasetName + ".png");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
