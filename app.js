var express = require('express');
var app = express();
var path = require('path');

//Route requirements
//var checkDate = require('./api/routes/checkDate');

app.use('/static', express.static(__dirname + '/static'));
app.use('/public', express.static(__dirname + '/public'));
app.use('/cryptopython', express.static(__dirname + '/cryptopython'));
app.use('/cryptopython/models', express.static(__dirname + '/cryptopython/models'))
//Setup logging (morgan) and body parsing
app.use(express.urlencoded({extended: false}));
app.use(express.json());

app.get('/', function(request, response) {
    response.sendFile(path.join(__dirname, 'index.html'));
});

app.get('/train', trainModel);
app.get('/test', testModel);

function testModel(req, res){
    var spawn = require("child_process").spawn;

    // E.g : http://localhost:80/test?model=modelname
    var process = spawn('python',["./cryptopython/lstm_predict_node.py",
        req.query.model] );

    process.stdout.on('data', function(data) {
        console.log(data.toString());
    });
    process.on('exit', function(data){
        res.send("done");
    });
}
function trainModel(req, res) {
    console.log("Training Model");
    // Use child_process.spawn method from child_process module and assign it to variable spawn
    const spawn = require("child_process").spawn;

    // E.g : http://localhost:80/train?hours=4000&seq_len=50&dropout=0.2&epochs=100&batch_size=50/
    var process = spawn('python',["./cryptopython/lstm_train_node.py",
        req.query.hours,
        req.query.seq_len,
        req.query.dropout,
        req.query.epochs,
        req.query.batch_size]);
    // Takes stdout data from script which executed with arguments and send this data to res object
    process.stdout.on('data', function(data) {
        console.log(data.toString());
    });
    process.stderr.on('data', function(data){
        console.log(data.toString());
    });
    process.on('exit', function(data){
        res.send("done");
    })

};

app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', '*');
    if (req.method === 'OPTIONS'){
        res.header('Access-Control-Allow-Methods', 'PUT, POST, PATCH, DELETE, GET, UPDATE');
        return res.status(200).json({});
    }
    next();
});

// Routing
var crypto_data = require('./cryptopython/stats.json');//Prevent CORS errors
app.get('/', function(request, response) {
    response.sendFile(path.join(__dirname, 'index.html'));
});

//routes that handle requests
//app.use('/api/checkDate', checkDate);

//Handle requests that didn't meet above routes
app.use((req, res, next) => {
    const error = new Error('Not found');
    error.status = 404;
    next(error);
});

app.use((error, req, res, next) => {
    res.status(error.status || 500);
    res.json({
        error: {
            message: error.message
        }
    });
});

module.exports = app;