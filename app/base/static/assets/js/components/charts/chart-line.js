'use strict';

//
// Sales chart
//

var SalesChart = (function() {

  // Variables

  var $chart = $('#chart-sales-dark');


  // Methods

  function init($chart) {

    var salesChart = new Chart($chart, {
      type: 'line',
      options: {
        scales: {
          yAxes: [{
            gridLines: {
              lineWidth: 1,
              color: Charts.colors.gray[900],
              zeroLineColor: Charts.colors.gray[900]
            },
            ticks: {
              callback: function(value) {
                if (!(value % 10)) {
                  return '$' + value + 'm';
                }
              }
            }
          }]
        },
        tooltips: {
          callbacks: {
            label: function(item, data) {
              var label = data.datasets[item.datasetIndex].label || '';
              var yLabel = item.yLabel;
              var content = '';

              if (data.datasets.length > 1) {
                content += '<span class="popover-body-label mr-auto">' + label + '</span>';
              }

              content += '<span class="popover-body-value">$' + yLabel + 'm</span>';
              return content;
            }
          }
        }
      },
      data: {
        labels: ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016','2017'],
        datasets: [{
          label: 'Performance',
          data: [9666, 12042, 14102, 17115, 19091, 19496, 19963, 20549, 22041]
        }]
      }
    });

    // Save to jQuery object

    $chart.data('chart', salesChart);

  };


  // Events

  if ($chart.length) {
    init($chart);
  }

})();
