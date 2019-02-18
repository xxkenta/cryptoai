var app = require('./app');
var server = require('http').createServer(app);
var os = require('os');

// Start server
var port = process.env.PORT || 80;
server.listen(port, function () {
    console.log('Express server listening on port %d in %s mode', port, app.get('env'))
});


