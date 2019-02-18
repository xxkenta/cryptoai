var checkRefresh = angular.module('checkRefresh', []);

function mainController($scope, $http){
    $scope.formData = {};
    $scope.refreshInfo = {};

    $scope.checkServiceTag = function(){
        $http.post('/api/checkDate/serviceTagLookup', $scope.formData)
            .success(function(data){
                $scope.formData = {};
                $scope.refreshInfo = data;
                console.log(JSON.stringify(data));
            })
            .error(function(data){
                console.log('Error: ' + JSON.stringify(data));
            })
    }
}