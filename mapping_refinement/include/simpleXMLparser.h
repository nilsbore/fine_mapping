#ifndef __SIMPLE_XML_PARSER__H
#define __SIMPLE_XML_PARSER__H

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include "ros/time.h"
#include "ros/serialization.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include "tf/tf.h"

#include <QFile>
#include <QDir>
#include <QXmlStreamWriter>

template <class PointType>
class SimpleXMLParser {
public:

    typedef pcl::PointCloud<PointType> Cloud;
    typedef typename Cloud::Ptr CloudPtr;

    struct RoomData
    {
        std::vector<CloudPtr>                            vIntermediateRoomClouds;
        std::vector<tf::StampedTransform>                vIntermediateRoomCloudTransforms;
        CloudPtr                                         completeRoomCloud;

        RoomData(){
            completeRoomCloud = CloudPtr(new Cloud());
        }
    };



    SimpleXMLParser()
    {
    }

    ~SimpleXMLParser()
    {

    }

    static RoomData loadRoomFromXML(const std::string& xmlFile)
    {
        RoomData aRoom;


        QFile file(xmlFile.c_str());

        if (!file.exists())
        {
            std::cerr<<"Could not open file "<<xmlFile<<" to load room."<<std::endl;
            return aRoom;
        }

        file.open(QIODevice::ReadOnly);

        QXmlStreamReader* xmlReader = new QXmlStreamReader(&file);
        Eigen::Vector4f centroid(0.0,0.0,0.0,0.0);

        while (!xmlReader->atEnd() && !xmlReader->hasError())
        {
            QXmlStreamReader::TokenType token = xmlReader->readNext();
            if (token == QXmlStreamReader::StartDocument)
                continue;

            if (xmlReader->hasError())
            {
                std::cout << "XML error: " << xmlReader->errorString().toStdString() << std::endl;
                return aRoom;
            }

            QString elementName = xmlReader->name().toString();

            if (token == QXmlStreamReader::StartElement)
            {
                if (xmlReader->name() == "RoomCompleteCloud")
                {
                    QXmlStreamAttributes attributes = xmlReader->attributes();
                    if (attributes.hasAttribute("filename"))
                    {
                        QString roomCompleteCloudFile = attributes.value("filename").toString();

                        int lastIndex = QString(xmlFile.c_str()).lastIndexOf("/");
                        roomCompleteCloudFile = QString(xmlFile.c_str()).left(lastIndex+1) + roomCompleteCloudFile;

                        std::cout<<"Loading complete cloud file name "<<roomCompleteCloudFile.toStdString()<<std::endl;
                        pcl::PCDReader reader;
                        CloudPtr cloud (new Cloud);
                        reader.read (roomCompleteCloudFile.toStdString(), *cloud);
                        *aRoom.completeRoomCloud = *cloud;
                    } else {
                        std::cerr<<"RoomCompleteCloud xml node does not have filename attribute. Aborting."<<std::endl;
                        return aRoom;
                    }
                }


                if (xmlReader->name() == "RoomIntermediateCloud")
                {
                    std::pair<std::string, tf::StampedTransform> cloudAndTransform = parseRoomIntermediateCloudNode(*xmlReader);
                    QString cloudFileName = cloudAndTransform.first.c_str();

                    int lastIndex = QString(xmlFile.c_str()).lastIndexOf("/");
                    cloudFileName = QString(xmlFile.c_str()).left(lastIndex+1) + cloudFileName;

                    std::cout<<"Loading intermediate cloud file name "<<cloudFileName.toStdString()<<std::endl;
                    pcl::PCDReader reader;
                    CloudPtr cloud (new Cloud);
                    reader.read (cloudFileName.toStdString(), *cloud);
                    aRoom.vIntermediateRoomClouds.push_back(cloud);
                    aRoom.vIntermediateRoomCloudTransforms.push_back(cloudAndTransform.second);

                }
            }
        }

        delete xmlReader;


        return aRoom;
    }

private:

    static std::pair<std::string, tf::StampedTransform> parseRoomIntermediateCloudNode(QXmlStreamReader& xmlReader)
    {
        tf::StampedTransform transform;
        geometry_msgs::TransformStamped tfmsg;
        std::pair<std::string, tf::StampedTransform> toRet;
        //        toRet.first = CloudPtr(new Cloud);
        QString intermediateParentNode("");

        if (xmlReader.name()!="RoomIntermediateCloud")
        {
            std::cerr<<"Cannot parse RoomIntermediateCloud node, it has a different name: "<<xmlReader.name().toString().toStdString()<<std::endl;
        }
        QXmlStreamAttributes attributes = xmlReader.attributes();
        if (attributes.hasAttribute("filename"))
        {
            QString roomIntermediateCloudFile = attributes.value("filename").toString();
            toRet.first = roomIntermediateCloudFile.toStdString();

        } else {
            std::cerr<<"RoomIntermediateCloud xml node does not have filename attribute. Aborting."<<std::endl;
            return toRet;
        }
        QXmlStreamReader::TokenType token = xmlReader.readNext();

        while(!((token == QXmlStreamReader::EndElement) && (xmlReader.name() == "RoomIntermediateCloud")) )
        {

            if (token == QXmlStreamReader::StartElement)
            {
                if (xmlReader.name() == "sec")
                {
                    int sec = xmlReader.readElementText().toInt();
                    tfmsg.header.stamp.sec = sec;
                }
                if (xmlReader.name() == "nsec")
                {
                    int nsec = xmlReader.readElementText().toInt();
                    tfmsg.header.stamp.nsec = nsec;
                }
                if (xmlReader.name() == "Translation")
                {
                    intermediateParentNode = xmlReader.name().toString();
                }
                if (xmlReader.name() == "Rotation")
                {
                    intermediateParentNode = xmlReader.name().toString();
                }
                if (xmlReader.name() == "w")
                {
                    double w = xmlReader.readElementText().toDouble();
                    tfmsg.transform.rotation.w = w;
                }
                if (xmlReader.name() == "x")
                {
                    double x = xmlReader.readElementText().toDouble();
                    if (intermediateParentNode == "Rotation")
                    {
                        tfmsg.transform.rotation.x = x;
                    }
                    if (intermediateParentNode == "Translation")
                    {
                        tfmsg.transform.translation.x = x;
                    }
                }
                if (xmlReader.name() == "y")
                {
                    double y = xmlReader.readElementText().toDouble();
                    if (intermediateParentNode == "Rotation")
                    {
                        tfmsg.transform.rotation.y = y;
                    }
                    if (intermediateParentNode == "Translation")
                    {
                        tfmsg.transform.translation.y = y;
                    }
                }
                if (xmlReader.name() == "z")
                {
                    double z = xmlReader.readElementText().toDouble();
                    if (intermediateParentNode == "Rotation")
                    {
                        tfmsg.transform.rotation.z = z;
                    }
                    if (intermediateParentNode == "Translation")
                    {
                        tfmsg.transform.translation.z = z;
                    }
                }
            }
            token = xmlReader.readNext();
        }

        tf::transformStampedMsgToTF(tfmsg, transform);
        toRet.second = transform;
        return toRet;
    }
};
#endif
